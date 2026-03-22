/**
 * AVBD 3D Physics Engine — Public API
 * Rapier-compatible API for 3D rigid body simulation.
 */

import type { Vec3, Quat, SolverConfig, RigidBodyHandle } from '../core/types.js';
import { RigidBodyType, DEFAULT_SOLVER_CONFIG_3D } from '../core/types.js';
import { AVBDSolver3D } from '../core/solver-3d.js';
import { RigidBodyDesc3D, ColliderDesc3D, type Body3D } from '../core/rigid-body-3d.js';
import { quatIdentity } from '../core/math.js';

export interface WorldConfig3D {
  iterations?: number;
  beta?: number;
  alpha?: number;
  gamma?: number;
  postStabilize?: boolean;
  dt?: number;
}

export class World3D {
  private solver: AVBDSolver3D;
  private bodyHandleMap: Map<number, RigidBody3D> = new Map();

  constructor(gravity: Vec3, config: WorldConfig3D = {}) {
    this.solver = new AVBDSolver3D({
      gravity,
      iterations: config.iterations ?? DEFAULT_SOLVER_CONFIG_3D.iterations,
      beta: config.beta ?? DEFAULT_SOLVER_CONFIG_3D.beta,
      alpha: config.alpha ?? DEFAULT_SOLVER_CONFIG_3D.alpha,
      gamma: config.gamma ?? DEFAULT_SOLVER_CONFIG_3D.gamma,
      postStabilize: config.postStabilize ?? DEFAULT_SOLVER_CONFIG_3D.postStabilize,
      dt: config.dt ?? DEFAULT_SOLVER_CONFIG_3D.dt,
    });
  }

  createRigidBody(desc: RigidBodyDesc3D): RigidBody3D {
    const handle = this.solver.bodyStore.addBody(desc);
    const rb = new RigidBody3D(handle, this.solver);
    this.bodyHandleMap.set(handle.index, rb);
    return rb;
  }

  createCollider(desc: ColliderDesc3D, body?: RigidBody3D): void {
    if (body) {
      this.solver.bodyStore.attachCollider(body.handle.index, desc);
    } else {
      const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
      const handle = this.solver.bodyStore.addBody(groundDesc);
      this.solver.bodyStore.attachCollider(handle.index, desc);
    }
  }

  step(): void {
    this.solver.step();
  }

  async stepAsync(): Promise<void> {
    this.step();
  }

  getBodyStates(): Float32Array {
    const bodies = this.solver.bodyStore.bodies;
    // 7 floats per body: x, y, z, qw, qx, qy, qz
    const data = new Float32Array(bodies.length * 7);
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      data[i * 7 + 0] = b.position.x;
      data[i * 7 + 1] = b.position.y;
      data[i * 7 + 2] = b.position.z;
      data[i * 7 + 3] = b.rotation.w;
      data[i * 7 + 4] = b.rotation.x;
      data[i * 7 + 5] = b.rotation.y;
      data[i * 7 + 6] = b.rotation.z;
    }
    return data;
  }

  removeRigidBody(body: RigidBody3D): void {
    const b = this.solver.bodyStore.bodies[body.handle.index];
    b.type = RigidBodyType.Fixed;
    b.mass = 0; b.invMass = 0;
    b.inertia = { x: 0, y: 0, z: 0 };
    b.invInertia = { x: 0, y: 0, z: 0 };
    b.halfExtents = { x: 0, y: 0, z: 0 };
    b.radius = 0;
    b.boundingRadius = 0;
    this.bodyHandleMap.delete(body.handle.index);
  }

  get numBodies(): number {
    return this.solver.bodyStore.bodies.filter(b => b.type === RigidBodyType.Dynamic).length;
  }

  get rawSolver(): AVBDSolver3D { return this.solver; }
}

export class RigidBody3D {
  readonly handle: RigidBodyHandle;
  private solver: AVBDSolver3D;

  constructor(handle: RigidBodyHandle, solver: AVBDSolver3D) {
    this.handle = handle;
    this.solver = solver;
  }

  private get body(): Body3D {
    return this.solver.bodyStore.bodies[this.handle.index];
  }

  translation(): Vec3 { return { ...this.body.position }; }
  rotation(): Quat { return { ...this.body.rotation }; }
  linvel(): Vec3 { return { ...this.body.velocity }; }
  angvel(): Vec3 { return { ...this.body.angularVelocity }; }
  mass(): number { return this.body.mass; }

  setTranslation(pos: Vec3): void { this.body.position = { ...pos }; }
  setRotation(q: Quat): void { this.body.rotation = { ...q }; }
  setLinvel(vel: Vec3): void { this.body.velocity = { ...vel }; }
  setAngvel(omega: Vec3): void { this.body.angularVelocity = { ...omega }; }

  applyImpulse(impulse: Vec3): void {
    if (this.body.invMass > 0) {
      this.body.velocity.x += impulse.x * this.body.invMass;
      this.body.velocity.y += impulse.y * this.body.invMass;
      this.body.velocity.z += impulse.z * this.body.invMass;
    }
  }

  applyForce(force: Vec3): void {
    const dt = this.solver.config.dt;
    if (this.body.invMass > 0) {
      this.body.velocity.x += force.x * this.body.invMass * dt;
      this.body.velocity.y += force.y * this.body.invMass * dt;
      this.body.velocity.z += force.z * this.body.invMass * dt;
    }
  }

  isDynamic(): boolean { return this.body.type === RigidBodyType.Dynamic; }
  isFixed(): boolean { return this.body.type === RigidBodyType.Fixed; }
}

// ─── Static AVBD3D namespace ────────────────────────────────────────────────

const AVBD3D = {
  async init(): Promise<void> {},
  World: World3D,
  RigidBodyDesc: RigidBodyDesc3D,
  ColliderDesc: ColliderDesc3D,
};

export default AVBD3D;
export { RigidBodyDesc3D, ColliderDesc3D };
