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
import { GPUSolver2D } from '../core/gpu-solver-2d.js';
import { GPUContext } from '../core/gpu-context.js';
import { RigidBodyDesc2D, ColliderDesc2D, type Body2D } from '../core/rigid-body.js';
import { JointData2D, createJointConstraintRows, type JointDef2D } from '../constraints/joint.js';
import { createSpringConstraintRows, type SpringDef2D } from '../constraints/spring.js';
import { MotorData2D, createMotorConstraintRows, type MotorDef2D } from '../constraints/motor.js';
import { ForceType } from '../core/types.js';

// ─── Module-level GPU state ─────────────────────────────────────────────────

let gpuContext: GPUContext | null = null;
let gpuAvailable = false;

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
  private gpuSolver: GPUSolver2D | null = null;
  private useGPU: boolean;
  private jointDefs: Map<number, JointDef2D> = new Map();
  private nextJointId = 0;
  private bodyHandleMap: Map<number, RigidBody> = new Map();
  private groundColliders: number[] = [];

  constructor(gravity: Vec2, config: WorldConfig = {}) {
    const solverConfig = {
      gravity,
      iterations: config.iterations ?? DEFAULT_SOLVER_CONFIG_2D.iterations,
      beta: config.beta ?? DEFAULT_SOLVER_CONFIG_2D.beta,
      alpha: config.alpha ?? DEFAULT_SOLVER_CONFIG_2D.alpha,
      gamma: config.gamma ?? DEFAULT_SOLVER_CONFIG_2D.gamma,
      postStabilize: config.postStabilize ?? DEFAULT_SOLVER_CONFIG_2D.postStabilize,
      dt: config.dt ?? DEFAULT_SOLVER_CONFIG_2D.dt,
    };

    // Always create CPU solver (used for step() and as fallback)
    this.solver = new AVBDSolver2D(solverConfig);

    // Create GPU solver if WebGPU was initialized
    this.useGPU = gpuAvailable && gpuContext !== null;
    if (this.useGPU && gpuContext) {
      this.gpuSolver = new GPUSolver2D(gpuContext, solverConfig);
    }
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

  /** Create a distance spring between two bodies */
  createSpring(
    bodyA: RigidBody, bodyB: RigidBody,
    anchorA: Vec2, anchorB: Vec2,
    restLength: number, stiffness: number, damping: number = 0,
  ): void {
    const def: SpringDef2D = {
      bodyA: bodyA.handle.index, bodyB: bodyB.handle.index,
      localAnchorA: anchorA, localAnchorB: anchorB,
      restLength, stiffness, damping,
    };
    const bA = this.solver.bodyStore.bodies[def.bodyA];
    const bB = this.solver.bodyStore.bodies[def.bodyB];
    const rows = createSpringConstraintRows(def, bA, bB, this.solver.config.penaltyMin);
    const indices = this.solver.constraintStore.addRows(rows);
    this.solver.jointConstraintIndices.push(...indices);
    // Add to ignore list
    const key = def.bodyA < def.bodyB
      ? `${def.bodyA}-${def.bodyB}` : `${def.bodyB}-${def.bodyA}`;
    this.solver.ignorePairs.add(key);
  }

  /** Create an angular velocity motor between two bodies */
  createMotor(
    bodyA: RigidBody, bodyB: RigidBody,
    targetVelocity: number, maxTorque: number = Infinity,
  ): void {
    const def: MotorDef2D = {
      bodyA: bodyA.handle.index, bodyB: bodyB.handle.index,
      targetVelocity, maxTorque, stiffness: Infinity,
    };
    const bA = this.solver.bodyStore.bodies[def.bodyA];
    const bB = this.solver.bodyStore.bodies[def.bodyB];
    const rows = createMotorConstraintRows(def, bA, bB, this.solver.config.penaltyMin, this.solver.config.dt);
    const indices = this.solver.constraintStore.addRows(rows);
    this.solver.jointConstraintIndices.push(...indices);
    const key = def.bodyA < def.bodyB
      ? `${def.bodyA}-${def.bodyB}` : `${def.bodyB}-${def.bodyA}`;
    this.solver.ignorePairs.add(key);
  }

  /**
   * Step the physics simulation on the GPU.
   * Dispatches WGSL compute shaders:
   * - Primal update: one dispatch per graph color group (parallel bodies)
   * - Dual update: one dispatch over all constraints
   * - Readback: async GPU→CPU transfer of body positions
   *
   * Requires AVBD.init() to have been called first.
   * Throws if GPU solver is not available.
   */
  async step(): Promise<void> {
    if (this.gpuSolver) {
      // Sync body/constraint stores from CPU World to GPU solver
      this.gpuSolver.bodyStore = this.solver.bodyStore;
      this.gpuSolver.constraintStore = this.solver.constraintStore;
      this.gpuSolver.ignorePairs = this.solver.ignorePairs;
      this.regenerateJointConstraints();
      await this.gpuSolver.step();
    } else {
      throw new Error(
        'GPU solver not available. Call AVBD.init() before creating a World, ' +
        'or use stepCPU() for explicit CPU fallback.'
      );
    }
  }

  /** @deprecated Use step() instead. */
  async stepAsync(): Promise<void> {
    await this.step();
  }

  /**
   * Step the physics simulation using the CPU solver (synchronous).
   * Opt-in only — use step() for GPU-accelerated physics.
   */
  stepCPU(): void {
    this.regenerateJointConstraints();
    this.solver.step();
  }

  /** Check if this world is using the GPU solver */
  get isGPU(): boolean {
    return this.gpuSolver !== null;
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
  /**
   * Initialize the AVBD engine with WebGPU.
   * Must be called before creating a World.
   * Throws if WebGPU is not available.
   */
  async init(): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error(
        'WebGPU is not available. AVBD requires a WebGPU-capable browser ' +
        '(Chrome 113+, Firefox Nightly, Safari 18+).'
      );
    }
    gpuContext = await GPUContext.create({ powerPreference: 'high-performance' });
    gpuAvailable = true;
  },

  /** Whether WebGPU GPU acceleration is available */
  get isGPUAvailable(): boolean {
    return gpuAvailable;
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
export { RigidBodyDesc2D, ColliderDesc2D, JointData2D, MotorData2D };
