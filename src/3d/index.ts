/**
 * AVBD 3D Physics Engine — Public API
 * GPU-first WebGPU-accelerated 6-DOF rigid body simulation.
 *
 * Usage:
 *   import AVBD3D from '@avbd/3d';
 *   await AVBD3D.init();  // Required: initializes WebGPU
 *   const world = new AVBD3D.World({ x: 0, y: -9.81, z: 0 });
 *   await world.step();   // GPU compute dispatch
 */

import type { Vec3, Quat, SolverConfig, RigidBodyHandle, StepTimings } from '../core/types.js';
import { RigidBodyType, DEFAULT_SOLVER_CONFIG_3D } from '../core/types.js';
import { GPUSolver3D } from '../core/gpu-solver-3d.js';
import { AVBDSolver3D } from '../core/solver-3d.js';
import { GPUContext } from '../core/gpu-context.js';
import { RigidBodyDesc3D, ColliderDesc3D, type Body3D } from '../core/rigid-body-3d.js';
import { quatIdentity } from '../core/math.js';

// ─── Module-level GPU state ─────────────────────────────────────────────────

let gpuContext: GPUContext | null = null;
let gpuAvailable = false;

export interface WorldConfig3D {
  iterations?: number;
  beta?: number;
  alpha?: number;
  gamma?: number;
  postStabilize?: boolean;
  dt?: number;
  /** Opt-in: use CPU solver instead of GPU. Default: false (GPU). */
  useCPU?: boolean;
  /** Use GPU-accelerated collision detection (default: true when GPU available) */
  useGPUCollision?: boolean;
}

export class World3D {
  private gpuSolver: GPUSolver3D | null = null;
  private cpuSolver: AVBDSolver3D | null = null;
  private bodyHandleMap: Map<number, RigidBody3D> = new Map();

  constructor(gravity: Vec3, config: WorldConfig3D = {}) {
    const solverConfig = {
      gravity,
      iterations: config.iterations ?? DEFAULT_SOLVER_CONFIG_3D.iterations,
      beta: config.beta ?? DEFAULT_SOLVER_CONFIG_3D.beta,
      alpha: config.alpha ?? DEFAULT_SOLVER_CONFIG_3D.alpha,
      gamma: config.gamma ?? DEFAULT_SOLVER_CONFIG_3D.gamma,
      postStabilize: config.postStabilize ?? DEFAULT_SOLVER_CONFIG_3D.postStabilize,
      dt: config.dt ?? DEFAULT_SOLVER_CONFIG_3D.dt,
      useGPUCollision: config.useGPUCollision,
    };

    if (config.useCPU) {
      this.cpuSolver = new AVBDSolver3D(solverConfig);
    } else {
      if (!gpuAvailable || !gpuContext) {
        throw new Error(
          'AVBD3D.init() must be called before creating a World3D. ' +
          'WebGPU is required. Pass { useCPU: true } for CPU-only mode.'
        );
      }
      this.gpuSolver = new GPUSolver3D(gpuContext, solverConfig);
    }
  }

  private get bodyStore() {
    return this.gpuSolver?.bodyStore ?? this.cpuSolver!.bodyStore;
  }

  private get activeSolver(): GPUSolver3D | AVBDSolver3D {
    return (this.gpuSolver ?? this.cpuSolver)!;
  }

  createRigidBody(desc: RigidBodyDesc3D): RigidBody3D {
    const handle = this.bodyStore.addBody(desc);
    const rb = new RigidBody3D(handle, this.activeSolver);
    this.bodyHandleMap.set(handle.index, rb);
    return rb;
  }

  createCollider(desc: ColliderDesc3D, body?: RigidBody3D): void {
    if (body) {
      this.bodyStore.attachCollider(body.handle.index, desc);
    } else {
      const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
      const handle = this.bodyStore.addBody(groundDesc);
      this.bodyStore.attachCollider(handle.index, desc);
    }
  }

  /**
   * Step the physics simulation.
   * Uses GPU compute shaders when available, CPU when useCPU was set.
   */
  step(): void | Promise<void> {
    if (this.gpuSolver) {
      return this.gpuSolver.step();
    }
    this.cpuSolver!.step();
  }

  /** @deprecated Use step() instead. */
  async stepAsync(): Promise<void> {
    await this.step();
  }

  /** Check if this world is using the GPU solver */
  get isGPU(): boolean {
    return this.gpuSolver !== null;
  }

  /** Per-step performance breakdown (ms) from the last step call */
  get lastTimings(): StepTimings | null {
    return this.gpuSolver?.lastTimings ?? this.cpuSolver?.lastTimings ?? null;
  }

  getBodyStates(): Float32Array {
    const bodies = this.bodyStore.bodies;
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
    const b = this.bodyStore.bodies[body.handle.index];
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
    return this.bodyStore.bodies.filter(b => b.type === RigidBodyType.Dynamic).length;
  }

  get rawSolver(): GPUSolver3D | AVBDSolver3D { return this.activeSolver; }
}

export class RigidBody3D {
  readonly handle: RigidBodyHandle;
  private solver: GPUSolver3D | AVBDSolver3D;

  constructor(handle: RigidBodyHandle, solver: GPUSolver3D | AVBDSolver3D) {
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
  /**
   * Initialize the AVBD 3D engine with WebGPU.
   * Must be called before creating a World3D.
   * Throws if WebGPU is not available.
   */
  /** Optional error callback for GPU device loss and other async errors */
  onError: null as ((message: string) => void) | null,

  async init(): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error(
        'WebGPU is not available. AVBD3D requires a WebGPU-capable browser ' +
        '(Chrome 113+, Firefox Nightly, Safari 18+).'
      );
    }
    gpuContext = await GPUContext.create({ powerPreference: 'high-performance' });
    gpuContext.onDeviceLost = (msg) => {
      AVBD3D.onError?.('GPU device lost: ' + msg);
    };
    gpuAvailable = true;
  },

  /** Whether WebGPU GPU acceleration is available */
  get isGPUAvailable(): boolean {
    return gpuAvailable;
  },

  World: World3D,
  RigidBodyDesc: RigidBodyDesc3D,
  ColliderDesc: ColliderDesc3D,
};

export default AVBD3D;
export { RigidBodyDesc3D, ColliderDesc3D };
export type { StepTimings };
