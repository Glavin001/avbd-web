/**
 * GPU-accelerated AVBD 3D solver.
 *
 * Dispatches WGSL compute shaders via WebGPU for the primal (6x6 LDL^T) and dual updates.
 * This is the primary solver path for 3D — GPU-first, no CPU fallback.
 *
 * Data flow per step:
 * 1. CPU: broadphase + narrowphase collision detection
 * 2. CPU: graph coloring for conflict-free parallel dispatch
 * 3. CPU→GPU: upload body state (20 floats/body), prev state (14 floats/body),
 *    sorted constraints (28 floats/row), color groups
 * 4. GPU: for each solver iteration:
 *    a. For each color group: dispatch 3D primal update shader (6-DOF per body)
 *    b. Dispatch 3D dual update shader (all constraints in one pass)
 * 5. GPU→CPU: async readback of body positions/quaternions + constraint lambdas
 * 6. CPU: velocity/angular velocity recovery, contact caching
 */

import type { Vec3, Quat, SolverConfig, ColorGroup, StepTimings } from './types.js';
import { RigidBodyType, DEFAULT_SOLVER_CONFIG_3D, COLLISION_MARGIN } from './types.js';
import { ForceType } from './types.js';
import type { Body3D } from './rigid-body-3d.js';
import { BodyStore3D } from './rigid-body-3d.js';
import type { ConstraintRow3D } from './solver-3d.js';
import { collide3D, getAABB3D, aabb3DOverlap } from '../3d/collision-gjk.js';
import { vec3Sub, vec3Scale, vec3Cross, vec3Dot, vec3Length, vec3, quatMul, quatNormalize, quatFromAxisAngle } from './math.js';
import { computeGraphColoring } from './graph-coloring.js';
import { GPUContext } from './gpu-context.js';
import { PRIMAL_UPDATE_3D_WGSL, DUAL_UPDATE_3D_WGSL, FRICTION_COUPLING_3D_WGSL } from '../shaders/embedded.js';

// ─── GPU Buffer Layout Constants ────────────────────────────────────────────

/** Body state 3D: [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz, mass, Ix,Iy,Iz, pad,pad,pad] = 20 floats */
const BODY_STRIDE = 20;
/** Body prev 3D: [px,py,pz, pqw,pqx,pqy,pqz, ix,iy,iz, iqw,iqx,iqy,iqz] = 14 floats */
const BODY_PREV_STRIDE = 14;
/**
 * ConstraintRow3D: matches WGSL struct exactly = 36 floats (144 bytes)
 * [body_a(i32), body_b(i32), force_type(u32), _pad(u32),  // 4 = 16 bytes
 *  jacobian_a_lin(vec4), jacobian_a_ang(vec4),              // 2 vec4 = 32 bytes
 *  jacobian_b_lin(vec4), jacobian_b_ang(vec4),              // 2 vec4 = 32 bytes
 *  hessian_diag_a_ang(vec4), hessian_diag_b_ang(vec4),      // 2 vec4 = 32 bytes
 *  c, c0, lambda, penalty, stiffness, fmin, fmax, active]  // 8 = 32 bytes
 * Total: 144 bytes
 */
const CONSTRAINT_STRIDE = 36;
/**
 * SolverParams 3D: 12 fields = 48 bytes
 * [dt, gravity_x, gravity_y, gravity_z, penalty_min, penalty_max, beta, alpha,
 *  num_bodies(u32), num_constraints(u32), num_bodies_in_group(u32), is_stabilization(u32)]
 */
const SOLVER_PARAMS_FLOATS = 12;
const SOLVER_PARAMS_BYTES = SOLVER_PARAMS_FLOATS * 4;
const WORKGROUP_SIZE = 64;

/** Helper to write typed arrays to GPU buffers */
function gpuWrite(queue: GPUQueue, buffer: GPUBuffer, offset: number, data: { buffer: ArrayBufferLike; byteOffset: number; byteLength: number }): void {
  (queue as any).writeBuffer(buffer, offset, data);
}

// ─── 3D Contact Constraint Creation ─────────────────────────────────────────

function createDefaultRow3D(): ConstraintRow3D {
  return {
    bodyA: -1, bodyB: -1,
    type: ForceType.Contact,
    jacobianA: [0, 0, 0, 0, 0, 0],
    jacobianB: [0, 0, 0, 0, 0, 0],
    hessianDiagA: [0, 0, 0, 0, 0, 0],
    hessianDiagB: [0, 0, 0, 0, 0, 0],
    c: 0, c0: 0,
    lambda: 0, penalty: 100,
    stiffness: Infinity,
    fmin: -Infinity, fmax: Infinity,
    active: true, broken: false,
  };
}

function computeTangent(n: Vec3): Vec3 {
  const up = Math.abs(n.y) < 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0);
  const t = vec3Cross(n, up);
  const len = vec3Length(t);
  return len > 1e-10 ? vec3Scale(t, 1 / len) : vec3(1, 0, 0);
}

// ─── GPU Solver ─────────────────────────────────────────────────────────────

export class GPUSolver3D {
  config: SolverConfig;
  bodyStore: BodyStore3D;
  constraintRows: ConstraintRow3D[] = [];
  ignorePairs: Set<string> = new Set();
  jointRows: ConstraintRow3D[] = [];
  colorGroups: ColorGroup[] = [];
  lastTimings: StepTimings | null = null;

  /** Contact cache for warmstarting between frames (body pair key → cached lambdas/penalties) */
  private contactCache: Map<string, { normalLambda: number; normalPenalty: number; fric1Lambda: number; fric1Penalty: number; fric2Lambda: number; fric2Penalty: number; age: number }> = new Map();

  private gpu: GPUContext;
  private initialized = false;

  // GPU resources
  private bodyStateBuffer!: GPUBuffer;
  private bodyPrevBuffer!: GPUBuffer;
  private constraintBuffer!: GPUBuffer;
  private solverParamsBuffer!: GPUBuffer;
  private colorIndicesBuffer!: GPUBuffer;
  private bodyConstraintRangesBuffer!: GPUBuffer;
  private constraintIndicesBuffer!: GPUBuffer;

  private primalPipeline!: GPUComputePipeline;
  private dualPipeline!: GPUComputePipeline;
  private frictionPipeline!: GPUComputePipeline;
  private primalBindGroupLayout!: GPUBindGroupLayout;
  private dualBindGroupLayout!: GPUBindGroupLayout;
  private frictionBindGroupLayout!: GPUBindGroupLayout;
  private frictionParamsBuffer!: GPUBuffer;

  private maxBodies = 0;
  private maxConstraints = 0;

  constructor(gpu: GPUContext, config: Partial<SolverConfig> = {}) {
    this.gpu = gpu;
    this.config = { ...DEFAULT_SOLVER_CONFIG_3D, ...config };
    this.bodyStore = new BodyStore3D();
  }

  /** Initialize GPU pipelines. Called automatically on first step. */
  init(): void {
    if (this.initialized) return;
    const device = this.gpu.device;

    // Primal pipeline: 7 bindings
    this.primalBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // body_state (read_write)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_prev
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // constraints
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // color_body_indices
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_constraint_ranges
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // constraint_indices
      ],
    });

    const primalModule = device.createShaderModule({ code: PRIMAL_UPDATE_3D_WGSL });
    this.primalPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.primalBindGroupLayout] }),
      compute: { module: primalModule, entryPoint: 'main' },
    });

    // Dual pipeline: 4 bindings
    this.dualBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_state (read)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_prev
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // constraints (read_write)
      ],
    });

    const dualModule = device.createShaderModule({ code: DUAL_UPDATE_3D_WGSL });
    this.dualPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.dualBindGroupLayout] }),
      compute: { module: dualModule, entryPoint: 'main' },
    });

    // Friction coupling pipeline: 2 bindings (params uniform + constraints read_write)
    this.frictionBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const frictionModule = device.createShaderModule({ code: FRICTION_COUPLING_3D_WGSL });
    this.frictionPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frictionBindGroupLayout] }),
      compute: { module: frictionModule, entryPoint: 'main' },
    });

    // Solver params uniform
    this.solverParamsBuffer = device.createBuffer({
      size: SOLVER_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Friction params uniform (4 bytes for num_constraints)
    this.frictionParamsBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.initialized = true;
  }

  /**
   * Run one physics timestep on the GPU.
   */
  async step(): Promise<void> {
    if (!this.initialized) this.init();

    const t0 = performance.now();
    const { config, bodyStore, gpu } = this;
    const dt = config.dt;
    const gravity = config.gravity as Vec3;
    const bodies = bodyStore.bodies;
    const device = gpu.device;

    if (bodies.length === 0) return;

    // ─── 1. CPU: Cache previous contacts, then collision detect ────
    // Cache lambda/penalty from previous frame's contact rows
    this.cacheContacts();
    this.constraintRows = [...this.jointRows];

    // Spatial hash broadphase: O(n) average instead of O(n²)
    const gpuCandidatePairs: [number, number][] = [];
    {
      const n = bodies.length;
      const aabbs = new Array(n);
      for (let i = 0; i < n; i++) aabbs[i] = getAABB3D(bodies[i]);

      let totalExtent = 0, dynCount = 0;
      for (let i = 0; i < n; i++) {
        if (bodies[i].type !== RigidBodyType.Fixed) {
          const a = aabbs[i];
          totalExtent += (a.max.x - a.min.x) + (a.max.y - a.min.y) + (a.max.z - a.min.z);
          dynCount++;
        }
      }
      const cellSize = Math.max(dynCount > 0 ? (totalExtent / (dynCount * 3)) * 2 : 1, 0.5);
      const invCell = 1 / cellSize;

      const grid = new Map<number, number[]>();
      function hashKey3D(cx: number, cy: number, cz: number): number {
        return ((cx + 0x400) * 0x100000) + ((cy + 0x400) * 0x800) + (cz + 0x400);
      }

      for (let i = 0; i < n; i++) {
        const a = aabbs[i];
        const x0 = Math.floor(a.min.x * invCell), x1 = Math.floor(a.max.x * invCell);
        const y0 = Math.floor(a.min.y * invCell), y1 = Math.floor(a.max.y * invCell);
        const z0 = Math.floor(a.min.z * invCell), z1 = Math.floor(a.max.z * invCell);
        for (let cx = x0; cx <= x1; cx++) {
          for (let cy = y0; cy <= y1; cy++) {
            for (let cz = z0; cz <= z1; cz++) {
              const k = hashKey3D(cx, cy, cz);
              let cell = grid.get(k);
              if (!cell) { cell = []; grid.set(k, cell); }
              cell.push(i);
            }
          }
        }
      }

      const tested = new Set<number>();
      for (const cell of grid.values()) {
        for (let ci = 0; ci < cell.length; ci++) {
          const i = cell[ci];
          for (let cj = ci + 1; cj < cell.length; cj++) {
            const j = cell[cj];
            const pk = i < j ? i * n + j : j * n + i;
            if (tested.has(pk)) continue;
            tested.add(pk);
            if (bodies[i].type === RigidBodyType.Fixed && bodies[j].type === RigidBodyType.Fixed) continue;
            const key = `${i}-${j}`;
            if (this.ignorePairs.has(key)) continue;
            if (!aabb3DOverlap(aabbs[i], aabbs[j])) continue;
            gpuCandidatePairs.push([i, j]);
          }
        }
      }
    }

    const tBroadphase = performance.now();

    // Narrowphase: GJK collision detection on candidate pairs
    for (const [i, j] of gpuCandidatePairs) {
      const manifold = collide3D(bodies[i], bodies[j]);
      if (manifold) {
        const rows = this.createContactRows3D(manifold, bodies[i], bodies[j]);
        this.constraintRows.push(...rows);
      }
    }

    const tCollision = performance.now();

    // ─── 2. CPU: Warmstart & Initialize ───────────────────────────
    // Apply cached warmstart values to new contact rows
    this.warmstartContacts();

    for (const row of this.constraintRows) {
      if (!row.active) continue;
      row.penalty *= config.gamma;
      row.penalty = Math.max(config.penaltyMin, Math.min(config.penaltyMax, row.penalty));
      if (row.penalty > row.stiffness) row.penalty = row.stiffness;
    }

    // Graph coloring
    const constraintPairs: [number, number][] = [];
    for (const row of this.constraintRows) {
      if (row.active && row.bodyA >= 0 && row.bodyB >= 0) {
        constraintPairs.push([row.bodyA, row.bodyB]);
      }
    }
    const fixedBodies = new Set<number>();
    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) fixedBodies.add(body.index);
    }
    this.colorGroups = computeGraphColoring(bodies.length, constraintPairs, fixedBodies);

    // Initialize bodies
    const gravMag = vec3Length(gravity);
    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;
      body.prevPosition = { ...body.position };
      body.prevRotation = { ...body.rotation };

      // Adaptive gravity weighting: only reduce gravity for slow-moving (supported)
      // bodies. Fast-moving bodies need full gravity to avoid artificial bounce on impact.
      let gravWeight = 1;
      if (gravMag > 0) {
        const speed = vec3Length(body.velocity);
        if (speed < 0.5) {
          const dvx = body.velocity.x - body.prevVelocity.x;
          const dvy = body.velocity.y - body.prevVelocity.y;
          const dvz = body.velocity.z - body.prevVelocity.z;
          const dvMag = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
          if (dvMag > 0.01) {
            const gravDir = { x: gravity.x / gravMag, y: gravity.y / gravMag, z: gravity.z / gravMag };
            const accelInGravDir = (dvx * gravDir.x + dvy * gravDir.y + dvz * gravDir.z) / dt;
            gravWeight = Math.max(0, Math.min(1, accelInGravDir / gravMag));
          }
        }
      }

      // Implicit angular damping
      const angDampFactor = 1 / (1 + 0.05 * dt);
      body.angularVelocity = vec3Scale(body.angularVelocity, angDampFactor);

      // Inertial target uses FULL gravity (the optimization objective target).
      body.inertialPosition = {
        x: body.prevPosition.x + body.velocity.x * dt + gravity.x * body.gravityScale * dt * dt,
        y: body.prevPosition.y + body.velocity.y * dt + gravity.y * body.gravityScale * dt * dt,
        z: body.prevPosition.z + body.velocity.z * dt + gravity.z * body.gravityScale * dt * dt,
      };
      const wLen = vec3Length(body.angularVelocity);
      if (wLen > 1e-10) {
        const axis = vec3Scale(body.angularVelocity, 1 / wLen);
        const dq = quatFromAxisAngle(axis, wLen * dt);
        body.inertialRotation = quatNormalize(quatMul(dq, body.prevRotation));
      } else {
        body.inertialRotation = { ...body.prevRotation };
      }

      // Move body to predicted position (initial guess for solver).
      body.position = {
        x: body.prevPosition.x + body.velocity.x * dt + gravity.x * body.gravityScale * gravWeight * dt * dt,
        y: body.prevPosition.y + body.velocity.y * dt + gravity.y * body.gravityScale * gravWeight * dt * dt,
        z: body.prevPosition.z + body.velocity.z * dt + gravity.z * body.gravityScale * gravWeight * dt * dt,
      };
      if (wLen > 1e-10) {
        const axis = vec3Scale(body.angularVelocity, 1 / wLen);
        const dq = quatFromAxisAngle(axis, wLen * dt);
        body.rotation = quatNormalize(quatMul(dq, body.prevRotation));
      }
      body.prevVelocity = { ...body.velocity };
    }

    const tInit = performance.now();

    // ─── 3. CPU→GPU: Upload Buffers ───────────────────────────────
    const numBodies = bodies.length;
    const numConstraints = this.constraintRows.length;

    // Build per-body constraint indirection (indices into original constraint array)
    const { bodyRanges, constraintIndices } = this.buildConstraintIndirection(numBodies, numConstraints);

    // Ensure GPU buffers are allocated BEFORE uploading data
    this.ensureBuffers(numBodies, Math.max(numConstraints, 1), constraintIndices.length);

    // Upload body state (20 floats per body)
    const bodyData = new Float32Array(numBodies * BODY_STRIDE);
    for (let i = 0; i < numBodies; i++) {
      const b = bodies[i];
      const off = i * BODY_STRIDE;
      bodyData[off + 0] = b.position.x;
      bodyData[off + 1] = b.position.y;
      bodyData[off + 2] = b.position.z;
      bodyData[off + 3] = b.rotation.w;
      bodyData[off + 4] = b.rotation.x;
      bodyData[off + 5] = b.rotation.y;
      bodyData[off + 6] = b.rotation.z;
      bodyData[off + 7] = b.velocity.x;
      bodyData[off + 8] = b.velocity.y;
      bodyData[off + 9] = b.velocity.z;
      bodyData[off + 10] = b.angularVelocity.x;
      bodyData[off + 11] = b.angularVelocity.y;
      bodyData[off + 12] = b.angularVelocity.z;
      bodyData[off + 13] = b.mass;
      bodyData[off + 14] = b.inertia.x;
      bodyData[off + 15] = b.inertia.y;
      bodyData[off + 16] = b.inertia.z;
      // [17..19] padding
    }
    gpuWrite(device.queue, this.bodyStateBuffer, 0, bodyData);

    // Upload prev/inertial state (14 floats per body)
    const prevData = new Float32Array(numBodies * BODY_PREV_STRIDE);
    for (let i = 0; i < numBodies; i++) {
      const b = bodies[i];
      const off = i * BODY_PREV_STRIDE;
      prevData[off + 0] = b.prevPosition.x;
      prevData[off + 1] = b.prevPosition.y;
      prevData[off + 2] = b.prevPosition.z;
      prevData[off + 3] = b.prevRotation.w;
      prevData[off + 4] = b.prevRotation.x;
      prevData[off + 5] = b.prevRotation.y;
      prevData[off + 6] = b.prevRotation.z;
      prevData[off + 7] = b.inertialPosition.x;
      prevData[off + 8] = b.inertialPosition.y;
      prevData[off + 9] = b.inertialPosition.z;
      prevData[off + 10] = b.inertialRotation.w;
      prevData[off + 11] = b.inertialRotation.x;
      prevData[off + 12] = b.inertialRotation.y;
      prevData[off + 13] = b.inertialRotation.z;
    }
    gpuWrite(device.queue, this.bodyPrevBuffer, 0, prevData);

    // Upload constraints in original order (28 floats = 112 bytes per row)
    if (numConstraints > 0) {
      const crData = new ArrayBuffer(numConstraints * CONSTRAINT_STRIDE * 4);
      const crView = new DataView(crData);
      for (let i = 0; i < numConstraints; i++) {
        const row = this.constraintRows[i];
        const byteOff = i * CONSTRAINT_STRIDE * 4;
        crView.setInt32(byteOff + 0, row.bodyA, true);
        crView.setInt32(byteOff + 4, row.bodyB, true);
        crView.setUint32(byteOff + 8, row.type, true);
        crView.setUint32(byteOff + 12, 0, true); // padding
        // jacobian_a_lin (vec4): 3 floats + padding
        crView.setFloat32(byteOff + 16, row.jacobianA[0], true);
        crView.setFloat32(byteOff + 20, row.jacobianA[1], true);
        crView.setFloat32(byteOff + 24, row.jacobianA[2], true);
        crView.setFloat32(byteOff + 28, 0, true);
        // jacobian_a_ang (vec4): 3 floats + padding
        crView.setFloat32(byteOff + 32, row.jacobianA[3], true);
        crView.setFloat32(byteOff + 36, row.jacobianA[4], true);
        crView.setFloat32(byteOff + 40, row.jacobianA[5], true);
        crView.setFloat32(byteOff + 44, 0, true);
        // jacobian_b_lin (vec4): 3 floats + padding
        crView.setFloat32(byteOff + 48, row.jacobianB[0], true);
        crView.setFloat32(byteOff + 52, row.jacobianB[1], true);
        crView.setFloat32(byteOff + 56, row.jacobianB[2], true);
        crView.setFloat32(byteOff + 60, 0, true);
        // jacobian_b_ang (vec4): 3 floats + .w stores friction coefficient mu
        crView.setFloat32(byteOff + 64, row.jacobianB[3], true);
        crView.setFloat32(byteOff + 68, row.jacobianB[4], true);
        crView.setFloat32(byteOff + 72, row.jacobianB[5], true);
        // Pack mu (friction coefficient) into jacobian_b_ang.w for GPU dual shader
        let mu = 0.5;
        if (row.type === ForceType.Contact && row.bodyA >= 0 && row.bodyB >= 0) {
          const bA = this.bodyStore.bodies[row.bodyA];
          const bB = this.bodyStore.bodies[row.bodyB];
          if (bA && bB) mu = Math.sqrt(bA.friction * bB.friction);
        }
        crView.setFloat32(byteOff + 76, mu, true);
        // hessian_diag_a_ang (vec4): angular components only (linear is always 0)
        const hA = row.hessianDiagA || [0, 0, 0, 0, 0, 0];
        crView.setFloat32(byteOff + 80, hA[3] || 0, true);
        crView.setFloat32(byteOff + 84, hA[4] || 0, true);
        crView.setFloat32(byteOff + 88, hA[5] || 0, true);
        crView.setFloat32(byteOff + 92, 0, true);
        // hessian_diag_b_ang (vec4)
        const hB = row.hessianDiagB || [0, 0, 0, 0, 0, 0];
        crView.setFloat32(byteOff + 96, hB[3] || 0, true);
        crView.setFloat32(byteOff + 100, hB[4] || 0, true);
        crView.setFloat32(byteOff + 104, hB[5] || 0, true);
        crView.setFloat32(byteOff + 108, 0, true);
        // scalar fields
        crView.setFloat32(byteOff + 112, row.c, true);
        crView.setFloat32(byteOff + 116, row.c0, true);
        crView.setFloat32(byteOff + 120, row.lambda, true);
        crView.setFloat32(byteOff + 124, row.penalty, true);
        crView.setFloat32(byteOff + 128, isFinite(row.stiffness) ? row.stiffness : 1e30, true);
        crView.setFloat32(byteOff + 132, isFinite(row.fmin) ? row.fmin : -1e30, true);
        crView.setFloat32(byteOff + 136, isFinite(row.fmax) ? row.fmax : 1e30, true);
        crView.setUint32(byteOff + 140, row.active ? 1 : 0, true);
      }
      gpuWrite(device.queue, this.constraintBuffer, 0, new Uint8Array(crData));
    }

    // Upload body constraint ranges and indirection indices
    gpuWrite(device.queue, this.bodyConstraintRangesBuffer, 0, bodyRanges);
    if (constraintIndices.length > 0) {
      gpuWrite(device.queue, this.constraintIndicesBuffer, 0, constraintIndices);
    }

    // ─── 4. GPU: Solver Iterations ────────────────────────────────
    const totalIterations = config.postStabilize ? config.iterations + 1 : config.iterations;
    const velocityRecoveryIter = config.iterations - 1;

    const tUpload = performance.now();

    // Push error scope to catch GPU validation errors
    device.pushErrorScope('validation');

    // Buffer to snapshot body state at velocity recovery iteration (before stabilization)
    const bodyReadbackSize = numBodies * BODY_STRIDE * 4;
    let velRecoveryBuffer: GPUBuffer | null = null;
    if (config.postStabilize) {
      velRecoveryBuffer = device.createBuffer({
        size: bodyReadbackSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    // Pre-write friction params (constant across iterations)
    if (numConstraints > 0) {
      gpuWrite(device.queue, this.frictionParamsBuffer, 0, new Uint32Array([numConstraints]));
    }

    for (let iter = 0; iter < totalIterations; iter++) {
      const isStabilization = config.postStabilize && iter === totalIterations - 1;

      // ─── 4a. Primal update: dispatch per color group ──────────
      // Each color group gets its own submit to ensure writeBuffer data
      // (params, indices) is correct for that dispatch. WebGPU writeBuffer
      // calls are resolved before the NEXT submit, so we must submit
      // after each group to avoid overwrites.
      for (const colorGroup of this.colorGroups) {
        if (colorGroup.bodyIndices.length === 0) continue;

        this.writeParams(dt, gravity, config, numBodies, numConstraints,
          colorGroup.bodyIndices.length, isStabilization);

        const colorData = new Uint32Array(colorGroup.bodyIndices);
        gpuWrite(device.queue, this.colorIndicesBuffer, 0, colorData);

        const bindGroup = device.createBindGroup({
          layout: this.primalBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.solverParamsBuffer } },
            { binding: 1, resource: { buffer: this.bodyStateBuffer } },
            { binding: 2, resource: { buffer: this.bodyPrevBuffer } },
            { binding: 3, resource: { buffer: this.constraintBuffer } },
            { binding: 4, resource: { buffer: this.colorIndicesBuffer } },
            { binding: 5, resource: { buffer: this.bodyConstraintRangesBuffer } },
            { binding: 6, resource: { buffer: this.constraintIndicesBuffer } },
          ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.primalPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(colorGroup.bodyIndices.length / WORKGROUP_SIZE));
        pass.end();
        device.queue.submit([encoder.finish()]);
      }

      // ─── 4b. Dual update + friction coupling ─────────────────
      if (!isStabilization && numConstraints > 0) {
        this.writeParams(dt, gravity, config, numBodies, numConstraints, 0, false);

        const encoder = device.createCommandEncoder();

        // Dual: update all constraint lambdas
        const dualBindGroup = device.createBindGroup({
          layout: this.dualBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.solverParamsBuffer } },
            { binding: 1, resource: { buffer: this.bodyStateBuffer } },
            { binding: 2, resource: { buffer: this.bodyPrevBuffer } },
            { binding: 3, resource: { buffer: this.constraintBuffer } },
          ],
        });

        const dualPass = encoder.beginComputePass();
        dualPass.setPipeline(this.dualPipeline);
        dualPass.setBindGroup(0, dualBindGroup);
        dualPass.dispatchWorkgroups(Math.ceil(numConstraints / WORKGROUP_SIZE));
        dualPass.end();

        // Friction coupling: separate pass after dual guarantees all
        // normal lambdas are finalized before updating friction bounds.
        // 3D contacts produce triplets: [normal, friction1, friction2].
        const numContactTriplets = Math.ceil(numConstraints / 3);
        const frictionBindGroup = device.createBindGroup({
          layout: this.frictionBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.frictionParamsBuffer } },
            { binding: 1, resource: { buffer: this.constraintBuffer } },
          ],
        });

        const fricPass = encoder.beginComputePass();
        fricPass.setPipeline(this.frictionPipeline);
        fricPass.setBindGroup(0, frictionBindGroup);
        fricPass.dispatchWorkgroups(Math.ceil(numContactTriplets / WORKGROUP_SIZE));
        fricPass.end();

        device.queue.submit([encoder.finish()]);
      }

      // Snapshot body state at velocity recovery iteration (before stabilization)
      if (iter === velocityRecoveryIter && velRecoveryBuffer) {
        const snapEncoder = device.createCommandEncoder();
        snapEncoder.copyBufferToBuffer(this.bodyStateBuffer, 0, velRecoveryBuffer, 0, bodyReadbackSize);
        device.queue.submit([snapEncoder.finish()]);
      }
    }

    // Check for GPU validation errors
    const validationError = await device.popErrorScope();
    if (validationError) {
      console.error('GPU validation error:', validationError.message);
    }

    const tDispatch = performance.now();

    // ─── 5. GPU→CPU: Read Back Results ────────────────────────────
    const bodyStagingBuffer = device.createBuffer({
      size: bodyReadbackSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(this.bodyStateBuffer, 0, bodyStagingBuffer, 0, bodyReadbackSize);

    let crStagingBuffer: GPUBuffer | null = null;
    if (numConstraints > 0) {
      const crReadbackSize = numConstraints * CONSTRAINT_STRIDE * 4;
      crStagingBuffer = device.createBuffer({
        size: crReadbackSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      copyEncoder.copyBufferToBuffer(this.constraintBuffer, 0, crStagingBuffer, 0, crReadbackSize);
    }

    device.queue.submit([copyEncoder.finish()]);

    // Map ALL staging buffers in parallel — single GPU fence wait instead of 3 serial ones
    const mapPromises: Promise<void>[] = [bodyStagingBuffer.mapAsync(GPUMapMode.READ)];
    if (velRecoveryBuffer) mapPromises.push(velRecoveryBuffer.mapAsync(GPUMapMode.READ));
    if (crStagingBuffer) mapPromises.push(crStagingBuffer.mapAsync(GPUMapMode.READ));
    await Promise.all(mapPromises);

    const bodyResult = new Float32Array(bodyStagingBuffer.getMappedRange().slice(0));
    bodyStagingBuffer.unmap();
    bodyStagingBuffer.destroy();

    let velRecoveryResult: Float32Array | null = null;
    if (velRecoveryBuffer) {
      velRecoveryResult = new Float32Array(velRecoveryBuffer.getMappedRange().slice(0));
      velRecoveryBuffer.unmap();
      velRecoveryBuffer.destroy();
    }

    // ─── 6. CPU: Apply Results ────────────────────────────────────
    // Use pre-stabilization positions for velocity recovery (matching CPU solver),
    // but final (post-stabilization) positions for body state
    const velSource = velRecoveryResult ?? bodyResult;

    for (let i = 0; i < numBodies; i++) {
      const body = bodies[i];
      if (body.type !== RigidBodyType.Dynamic) continue;

      const off = i * BODY_STRIDE;
      body.position.x = bodyResult[off + 0];
      body.position.y = bodyResult[off + 1];
      body.position.z = bodyResult[off + 2];
      body.rotation.w = bodyResult[off + 3];
      body.rotation.x = bodyResult[off + 4];
      body.rotation.y = bodyResult[off + 5];
      body.rotation.z = bodyResult[off + 6];

      // BDF1 velocity recovery from pre-stabilization positions
      const MAX_LIN_VEL = 100;
      const MAX_ANG_VEL = 50;
      const voff = i * BODY_STRIDE;
      body.velocity = {
        x: (velSource[voff + 0] - body.prevPosition.x) / dt,
        y: (velSource[voff + 1] - body.prevPosition.y) / dt,
        z: (velSource[voff + 2] - body.prevPosition.z) / dt,
      };
      // Clamp recovered linear velocity
      const vLen = vec3Length(body.velocity);
      if (vLen > MAX_LIN_VEL) {
        body.velocity = vec3Scale(body.velocity, MAX_LIN_VEL / vLen);
      }
      // Angular velocity from quaternion difference (using pre-stabilization rotation)
      const vqw = velSource[voff + 3];
      const vqx = velSource[voff + 4];
      const vqy = velSource[voff + 5];
      const vqz = velSource[voff + 6];
      const dq = quatMul(
        { w: vqw, x: vqx, y: vqy, z: vqz },
        {
          w: body.prevRotation.w,
          x: -body.prevRotation.x,
          y: -body.prevRotation.y,
          z: -body.prevRotation.z,
        },
      );
      body.angularVelocity = vec3Scale(vec3(dq.x, dq.y, dq.z), 2 / dt);
      // Clamp recovered angular velocity
      const wLen = vec3Length(body.angularVelocity);
      if (wLen > MAX_ANG_VEL) {
        body.angularVelocity = vec3Scale(body.angularVelocity, MAX_ANG_VEL / wLen);
      }
    }

    // Readback constraint lambdas for warmstarting (already mapped via Promise.all above)
    if (crStagingBuffer && numConstraints > 0) {
      const crResult = new DataView(crStagingBuffer.getMappedRange().slice(0));
      crStagingBuffer.unmap();
      crStagingBuffer.destroy();

      for (let i = 0; i < numConstraints; i++) {
        const byteOff = i * CONSTRAINT_STRIDE * 4;
        this.constraintRows[i].lambda = crResult.getFloat32(byteOff + 120, true);
        this.constraintRows[i].penalty = crResult.getFloat32(byteOff + 124, true);
      }
    }

    const tEnd = performance.now();
    this.lastTimings = {
      total: tEnd - t0,
      broadphase: tBroadphase - t0,
      narrowphase: tCollision - tBroadphase,
      warmstart: tInit - tCollision,
      bodyInit: 0,
      solverIters: 0,
      velocityRecover: 0,
      bufferUpload: tUpload - tInit,
      gpuDispatch: tDispatch - tUpload,
      readback: tEnd - tDispatch,
      numBodies: bodies.length,
      numConstraints: this.constraintRows.length,
    };
  }

  /** Create 3D contact constraint rows from a manifold */
  private createContactRows3D(
    manifold: { bodyA: number; bodyB: number; normal: Vec3; contacts: { position: Vec3; depth: number }[] },
    bodyA: Body3D,
    bodyB: Body3D,
  ): ConstraintRow3D[] {
    const rows: ConstraintRow3D[] = [];
    const mu = Math.sqrt(bodyA.friction * bodyB.friction);

    for (const contact of manifold.contacts) {
      const n = manifold.normal;
      const rA = vec3Sub(contact.position, bodyA.position);
      const rB = vec3Sub(contact.position, bodyB.position);

      // Normal constraint
      const nRow = createDefaultRow3D();
      nRow.bodyA = manifold.bodyA;
      nRow.bodyB = manifold.bodyB;
      nRow.type = ForceType.Contact;

      const torqueA = vec3Cross(rA, n);
      nRow.jacobianA = [n.x, n.y, n.z, torqueA.x, torqueA.y, torqueA.z];
      const torqueB = vec3Cross(rB, n);
      nRow.jacobianB = [-n.x, -n.y, -n.z, -torqueB.x, -torqueB.y, -torqueB.z];

      // Geometric stiffness (Hessian diagonal) for angular DOFs
      const rAdotN = vec3Dot(rA, n);
      const rBdotN = vec3Dot(rB, n);
      nRow.hessianDiagA = [0, 0, 0,
        -(rAdotN - rA.x * n.x),
        -(rAdotN - rA.y * n.y),
        -(rAdotN - rA.z * n.z)];
      nRow.hessianDiagB = [0, 0, 0,
        -(rBdotN - rB.x * n.x),
        -(rBdotN - rB.y * n.y),
        -(rBdotN - rB.z * n.z)];

      nRow.c = -contact.depth + COLLISION_MARGIN;
      nRow.c0 = nRow.c;
      nRow.fmin = -Infinity;
      nRow.fmax = 0;
      nRow.penalty = this.config.penaltyMin;
      rows.push(nRow);

      // Two friction tangent constraints
      const t1 = computeTangent(n);
      const t2 = vec3Cross(n, t1);
      const tLen = vec3Length(t2);
      const t2n = tLen > 1e-10 ? vec3Scale(t2, 1 / tLen) : vec3(0, 0, 1);

      for (const t of [t1, t2n]) {
        const fRow = createDefaultRow3D();
        fRow.bodyA = manifold.bodyA;
        fRow.bodyB = manifold.bodyB;
        fRow.type = ForceType.Contact;
        const tA = vec3Cross(rA, t);
        fRow.jacobianA = [t.x, t.y, t.z, tA.x, tA.y, tA.z];
        const tB = vec3Cross(rB, t);
        fRow.jacobianB = [-t.x, -t.y, -t.z, -tB.x, -tB.y, -tB.z];

        // Geometric stiffness for friction tangent
        const rAdotT = vec3Dot(rA, t);
        const rBdotT = vec3Dot(rB, t);
        fRow.hessianDiagA = [0, 0, 0,
          -(rAdotT - rA.x * t.x),
          -(rAdotT - rA.y * t.y),
          -(rAdotT - rA.z * t.z)];
        fRow.hessianDiagB = [0, 0, 0,
          -(rBdotT - rB.x * t.x),
          -(rBdotT - rB.y * t.y),
          -(rBdotT - rB.z * t.z)];

        fRow.c = 0;
        fRow.c0 = 0;
        fRow.fmin = -mu * this.config.penaltyMin * contact.depth;
        fRow.fmax = mu * this.config.penaltyMin * contact.depth;
        fRow.penalty = this.config.penaltyMin;
        rows.push(fRow);
      }
    }
    return rows;
  }

  /** Build per-body constraint index lists for indirection on GPU */
  private buildConstraintIndirection(numBodies: number, numConstraints: number): {
    bodyRanges: Uint32Array;
    constraintIndices: Uint32Array;
  } {
    const rows = this.constraintRows;

    // Build per-body constraint index lists
    const perBody: number[][] = Array.from({ length: numBodies }, () => []);
    for (let ci = 0; ci < numConstraints; ci++) {
      const row = rows[ci];
      if (!row.active) continue;
      if (row.bodyA >= 0 && row.bodyA < numBodies) perBody[row.bodyA].push(ci);
      if (row.bodyB >= 0 && row.bodyB < numBodies) perBody[row.bodyB].push(ci);
    }

    // Flatten into indirection array + build ranges
    const allIndices: number[] = [];
    const bodyRanges = new Uint32Array(numBodies * 2);
    for (let i = 0; i < numBodies; i++) {
      bodyRanges[i * 2 + 0] = allIndices.length;
      bodyRanges[i * 2 + 1] = perBody[i].length;
      allIndices.push(...perBody[i]);
    }

    return { bodyRanges, constraintIndices: new Uint32Array(allIndices) };
  }

  /** Write solver params uniform (3D: 12 fields) */
  private writeParams(
    dt: number, gravity: Vec3, config: SolverConfig,
    numBodies: number, numConstraints: number,
    numBodiesInGroup: number, isStabilization: boolean,
  ): void {
    const data = new ArrayBuffer(SOLVER_PARAMS_BYTES);
    const fv = new Float32Array(data);
    const uv = new Uint32Array(data);
    fv[0] = dt;
    fv[1] = gravity.x;
    fv[2] = gravity.y;
    fv[3] = gravity.z;
    fv[4] = config.penaltyMin;
    fv[5] = config.penaltyMax;
    fv[6] = config.beta;
    // Per-iteration alpha (reference: solver.cpp):
    // Normal iterations: alpha=1.0 → C0*(1-1)=0, only J·dp (position changes)
    // Stabilization: alpha=0.0 → C0*(1-0)=C0, full violation correction
    fv[7] = isStabilization ? 0.0 : 1.0;
    uv[8] = numBodies;
    uv[9] = numConstraints;
    uv[10] = numBodiesInGroup;
    uv[11] = isStabilization ? 1 : 0;
    gpuWrite(this.gpu.device.queue, this.solverParamsBuffer, 0, new Uint8Array(data));
  }

  private maxConstraintIndices = 0;

  /** Ensure GPU buffers are large enough */
  private ensureBuffers(numBodies: number, numConstraints: number, numConstraintIndices: number): void {
    const device = this.gpu.device;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    if (numBodies > this.maxBodies) {
      this.maxBodies = Math.max(numBodies, this.maxBodies * 2, 64);
      if (this.bodyStateBuffer) this.bodyStateBuffer.destroy();
      if (this.bodyPrevBuffer) this.bodyPrevBuffer.destroy();
      if (this.bodyConstraintRangesBuffer) this.bodyConstraintRangesBuffer.destroy();
      if (this.colorIndicesBuffer) this.colorIndicesBuffer.destroy();

      this.bodyStateBuffer = device.createBuffer({ size: this.maxBodies * BODY_STRIDE * 4, usage: STORAGE });
      this.bodyPrevBuffer = device.createBuffer({ size: this.maxBodies * BODY_PREV_STRIDE * 4, usage: STORAGE });
      this.bodyConstraintRangesBuffer = device.createBuffer({ size: this.maxBodies * 2 * 4, usage: STORAGE });
      this.colorIndicesBuffer = device.createBuffer({ size: this.maxBodies * 4, usage: STORAGE });
    }

    if (numConstraints > this.maxConstraints) {
      this.maxConstraints = Math.max(numConstraints, this.maxConstraints * 2, 64);
      if (this.constraintBuffer) this.constraintBuffer.destroy();
      this.constraintBuffer = device.createBuffer({ size: this.maxConstraints * CONSTRAINT_STRIDE * 4, usage: STORAGE });
    }

    const ciCount = Math.max(numConstraintIndices, 1);
    if (ciCount > this.maxConstraintIndices) {
      this.maxConstraintIndices = Math.max(ciCount, this.maxConstraintIndices * 2, 64);
      if (this.constraintIndicesBuffer) this.constraintIndicesBuffer.destroy();
      this.constraintIndicesBuffer = device.createBuffer({ size: this.maxConstraintIndices * 4, usage: STORAGE });
    }
  }

  /** Cache contact lambda/penalty from current frame for warmstarting next frame */
  private cacheContacts(): void {
    // Age existing entries
    for (const [key, cached] of this.contactCache) {
      cached.age++;
      if (cached.age > 5) this.contactCache.delete(key);
    }
    // Save current contacts (triplets: normal, fric1, fric2)
    const rows = this.constraintRows;
    for (let i = 0; i < rows.length; i += 3) {
      const row = rows[i];
      if (row.type !== ForceType.Contact) continue;
      if (i + 2 >= rows.length) break;
      const key = row.bodyA < row.bodyB ? `${row.bodyA}-${row.bodyB}` : `${row.bodyB}-${row.bodyA}`;
      this.contactCache.set(key, {
        normalLambda: row.lambda, normalPenalty: row.penalty,
        fric1Lambda: rows[i + 1].lambda, fric1Penalty: rows[i + 1].penalty,
        fric2Lambda: rows[i + 2].lambda, fric2Penalty: rows[i + 2].penalty,
        age: 0,
      });
    }
  }

  /** Apply cached warmstart values to newly created contact rows */
  private warmstartContacts(): void {
    const rows = this.constraintRows;
    for (let i = 0; i < rows.length; i += 3) {
      const row = rows[i];
      if (row.type !== ForceType.Contact) continue;
      if (i + 2 >= rows.length) break;
      const key = row.bodyA < row.bodyB ? `${row.bodyA}-${row.bodyB}` : `${row.bodyB}-${row.bodyA}`;
      const cached = this.contactCache.get(key);
      if (cached) {
        row.lambda = cached.normalLambda;
        row.penalty = cached.normalPenalty;
        rows[i + 1].lambda = cached.fric1Lambda;
        rows[i + 1].penalty = cached.fric1Penalty;
        rows[i + 2].lambda = cached.fric2Lambda;
        rows[i + 2].penalty = cached.fric2Penalty;
      }
    }
  }

  destroy(): void {
    for (const buf of [this.bodyStateBuffer, this.bodyPrevBuffer, this.constraintBuffer,
      this.solverParamsBuffer, this.frictionParamsBuffer, this.colorIndicesBuffer,
      this.bodyConstraintRangesBuffer, this.constraintIndicesBuffer]) {
      if (buf) buf.destroy();
    }
  }
}
