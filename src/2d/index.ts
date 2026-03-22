/**
 * AVBD 2D Physics Engine — Public API
 * Rapier-compatible API design for WebGPU-accelerated physics.
 *
 * Usage:
 *   import AVBD from '@avbd/2d';
 *   await AVBD.init();
 *   const world = new AVBD.World({ x: 0, y: -9.81 });
 *   ...
 */

import type { Vec2, SolverConfig, RigidBodyHandle, ContactManifold2D } from '../core/types.js';
import { RigidBodyType, DEFAULT_SOLVER_CONFIG_2D } from '../core/types.js';
import { AVBDSolver2D } from '../core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D, type Body2D } from '../core/rigid-body.js';
import { JointData2D, createJointConstraintRows, type JointDef2D } from '../constraints/joint.js';
import { ForceType } from '../core/types.js';

// ─── World ──────────────────────────────────────────────────────────────────

export interface WorldConfig {
  /** Solver iterations per timestep (default: 10) */
  iterations?: number;
  /** Penalty growth rate (default: 100000) */
  beta?: number;
  /** Stabilization parameter (default: 0.99) */
  alpha?: number;
  /** Warmstart decay (default: 0.99) */
  gamma?: number;
  /** Enable post-stabilization (default: true) */
  postStabilize?: boolean;
  /** Timestep (default: 1/60) */
  dt?: number;
}

export class World {
  private solver: AVBDSolver2D;
  private jointDefs: Map<number, JointDef2D> = new Map();
  private nextJointId = 0;
  private bodyHandleMap: Map<number, RigidBody> = new Map();
  private groundColliders: number[] = [];

  constructor(gravity: Vec2, config: WorldConfig = {}) {
    this.solver = new AVBDSolver2D({
      gravity,
      iterations: config.iterations ?? DEFAULT_SOLVER_CONFIG_2D.iterations,
      beta: config.beta ?? DEFAULT_SOLVER_CONFIG_2D.beta,
      alpha: config.alpha ?? DEFAULT_SOLVER_CONFIG_2D.alpha,
      gamma: config.gamma ?? DEFAULT_SOLVER_CONFIG_2D.gamma,
      postStabilize: config.postStabilize ?? DEFAULT_SOLVER_CONFIG_2D.postStabilize,
      dt: config.dt ?? DEFAULT_SOLVER_CONFIG_2D.dt,
    });
  }

  /** Create a rigid body in the world */
  createRigidBody(desc: RigidBodyDesc2D): RigidBody {
    const handle = this.solver.bodyStore.addBody(desc);
    const rb = new RigidBody(handle, this.solver);
    this.bodyHandleMap.set(handle.index, rb);
    return rb;
  }

  /** Attach a collider to a rigid body (or create a fixed body for ground) */
  createCollider(desc: ColliderDesc2D, body?: RigidBody): Collider {
    if (body) {
      this.solver.bodyStore.attachCollider(body.handle.index, desc);
      return new Collider(body.handle.index);
    } else {
      // Create a fixed body for the collider (ground plane, etc.)
      const groundDesc = RigidBodyDesc2D.fixed().setTranslation(0, 0);
      const handle = this.solver.bodyStore.addBody(groundDesc);
      this.solver.bodyStore.attachCollider(handle.index, desc);
      this.groundColliders.push(handle.index);
      return new Collider(handle.index);
    }
  }

  /** Create a joint between two rigid bodies */
  createJoint(params: JointData2D, bodyA: RigidBody, bodyB: RigidBody): JointHandle {
    const id = this.nextJointId++;
    const def: JointDef2D = {
      bodyA: bodyA.handle.index,
      bodyB: bodyB.handle.index,
      localAnchorA: params.localAnchorA,
      localAnchorB: params.localAnchorB,
      angleConstraint: params.angleConstraint,
      targetAngle: params.targetAngle,
      stiffness: params.stiffness,
      fractureThreshold: params.fractureThreshold,
    };
    this.jointDefs.set(id, def);

    // Add joint constraints to the store
    const bA = this.solver.bodyStore.bodies[def.bodyA];
    const bB = this.solver.bodyStore.bodies[def.bodyB];
    const rows = createJointConstraintRows(def, bA, bB, this.solver.config.penaltyMin);
    const indices = this.solver.constraintStore.addRows(rows);
    this.solver.jointConstraintIndices.push(...indices);

    // Add to ignore list so joint-connected bodies don't collide
    const key = def.bodyA < def.bodyB
      ? `${def.bodyA}-${def.bodyB}`
      : `${def.bodyB}-${def.bodyA}`;
    this.solver.ignorePairs.add(key);

    return new JointHandle(id);
  }

  /**
   * Step the physics simulation forward by one timestep.
   * For the CPU solver, this is synchronous.
   * For GPU solver (future), this will be async.
   */
  step(): void {
    // Regenerate joint constraints (Jacobians change with body positions)
    this.regenerateJointConstraints();

    this.solver.step();
  }

  /** Async step (for API compatibility with future GPU version) */
  async stepAsync(): Promise<void> {
    this.step();
  }

  /** Get all body states as a flat Float32Array [x, y, angle, ...] */
  getBodyStates(): Float32Array {
    const bodies = this.solver.bodyStore.bodies;
    const data = new Float32Array(bodies.length * 3);
    for (let i = 0; i < bodies.length; i++) {
      data[i * 3 + 0] = bodies[i].position.x;
      data[i * 3 + 1] = bodies[i].position.y;
      data[i * 3 + 2] = bodies[i].angle;
    }
    return data;
  }

  /** Remove a rigid body from the world */
  removeRigidBody(body: RigidBody): void {
    // Mark body as fixed with zero extents (effectively removes it)
    const b = this.solver.bodyStore.bodies[body.handle.index];
    b.type = RigidBodyType.Fixed;
    b.mass = 0;
    b.invMass = 0;
    b.inertia = 0;
    b.invInertia = 0;
    b.halfExtents = { x: 0, y: 0 };
    b.radius = 0;
    b.boundingRadius = 0;
    this.bodyHandleMap.delete(body.handle.index);
  }

  /** Get the number of dynamic bodies */
  get numBodies(): number {
    return this.solver.bodyStore.bodies.filter(
      b => b.type === RigidBodyType.Dynamic
    ).length;
  }

  /** Get the total number of active constraints */
  get numConstraints(): number {
    return this.solver.constraintStore.activeCount;
  }

  /** Get the underlying solver (for advanced usage) */
  get rawSolver(): AVBDSolver2D {
    return this.solver;
  }

  private regenerateJointConstraints(): void {
    // Remove old joint constraints
    this.solver.constraintStore.rows = this.solver.constraintStore.rows.filter(
      r => r.type !== ForceType.Joint
    );
    this.solver.jointConstraintIndices = [];

    // Recreate with current positions
    for (const [, def] of this.jointDefs) {
      const bA = this.solver.bodyStore.bodies[def.bodyA];
      const bB = this.solver.bodyStore.bodies[def.bodyB];
      const rows = createJointConstraintRows(def, bA, bB, this.solver.config.penaltyMin);
      const indices = this.solver.constraintStore.addRows(rows);
      this.solver.jointConstraintIndices.push(...indices);
    }
  }
}

// ─── RigidBody ──────────────────────────────────────────────────────────────

export class RigidBody {
  readonly handle: RigidBodyHandle;
  private solver: AVBDSolver2D;

  constructor(handle: RigidBodyHandle, solver: AVBDSolver2D) {
    this.handle = handle;
    this.solver = solver;
  }

  private get body(): Body2D {
    return this.solver.bodyStore.bodies[this.handle.index];
  }

  /** Get the body's translation (position) */
  translation(): Vec2 {
    return { ...this.body.position };
  }

  /** Get the body's rotation angle (radians) */
  rotation(): number {
    return this.body.angle;
  }

  /** Get the body's linear velocity */
  linvel(): Vec2 {
    return { ...this.body.velocity };
  }

  /** Get the body's angular velocity */
  angvel(): number {
    return this.body.angularVelocity;
  }

  /** Get the body's mass */
  mass(): number {
    return this.body.mass;
  }

  /** Set the body's translation */
  setTranslation(pos: Vec2, wakeUp: boolean = true): void {
    this.body.position = { ...pos };
  }

  /** Set the body's rotation */
  setRotation(angle: number, wakeUp: boolean = true): void {
    this.body.angle = angle;
  }

  /** Set the body's linear velocity */
  setLinvel(vel: Vec2, wakeUp: boolean = true): void {
    this.body.velocity = { ...vel };
  }

  /** Set the body's angular velocity */
  setAngvel(omega: number, wakeUp: boolean = true): void {
    this.body.angularVelocity = omega;
  }

  /** Apply an impulse (instantaneous velocity change) */
  applyImpulse(impulse: Vec2, wakeUp: boolean = true): void {
    if (this.body.invMass > 0) {
      this.body.velocity.x += impulse.x * this.body.invMass;
      this.body.velocity.y += impulse.y * this.body.invMass;
    }
  }

  /** Apply a torque impulse */
  applyTorqueImpulse(torque: number, wakeUp: boolean = true): void {
    if (this.body.invInertia > 0) {
      this.body.angularVelocity += torque * this.body.invInertia;
    }
  }

  /** Apply a force (accumulated over timestep) */
  applyForce(force: Vec2, wakeUp: boolean = true): void {
    const dt = this.solver.config.dt;
    if (this.body.invMass > 0) {
      this.body.velocity.x += force.x * this.body.invMass * dt;
      this.body.velocity.y += force.y * this.body.invMass * dt;
    }
  }

  /** Check if body is dynamic */
  isDynamic(): boolean {
    return this.body.type === RigidBodyType.Dynamic;
  }

  /** Check if body is fixed */
  isFixed(): boolean {
    return this.body.type === RigidBodyType.Fixed;
  }
}

// ─── Collider ───────────────────────────────────────────────────────────────

export class Collider {
  readonly bodyIndex: number;

  constructor(bodyIndex: number) {
    this.bodyIndex = bodyIndex;
  }
}

// ─── JointHandle ────────────────────────────────────────────────────────────

export class JointHandle {
  readonly id: number;

  constructor(id: number) {
    this.id = id;
  }
}

// ─── Static AVBD namespace (Rapier-style entry point) ───────────────────────

const AVBD = {
  /** Initialize the AVBD engine (required before creating a world) */
  async init(): Promise<void> {
    // CPU solver doesn't need async init, but we keep the API
    // compatible with future GPU version
  },

  /** World class */
  World,

  /** Rigid body description builder */
  RigidBodyDesc: RigidBodyDesc2D,

  /** Collider description builder */
  ColliderDesc: ColliderDesc2D,

  /** Joint data builder */
  JointData: JointData2D,
};

export default AVBD;
export { RigidBodyDesc2D, ColliderDesc2D, JointData2D };
