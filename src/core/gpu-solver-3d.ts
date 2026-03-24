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
import { BodyStore3D, ColliderShapeType3D } from './rigid-body-3d.js';
import type { ConstraintRow3D } from './solver-3d.js';
import { collide3D } from '../3d/collision-gjk.js';
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

  /** Guard against concurrent step() calls (async game loops can overlap) */
  private _stepping = false;

  /** Contact cache for warmstarting between frames (body pair key → cached lambdas/penalties) */
  private contactCache: Map<number, { normalLambda: number; normalPenalty: number; fric1Lambda: number; fric1Penalty: number; fric2Lambda: number; fric2Penalty: number; age: number }> = new Map();

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

  // ─── Reusable buffers (avoid per-frame allocation) ───────────────────────
  // Broadphase flat AABB arrays
  private _aabbMinX: Float64Array = new Float64Array(0);
  private _aabbMinY: Float64Array = new Float64Array(0);
  private _aabbMinZ: Float64Array = new Float64Array(0);
  private _aabbMaxX: Float64Array = new Float64Array(0);
  private _aabbMaxY: Float64Array = new Float64Array(0);
  private _aabbMaxZ: Float64Array = new Float64Array(0);
  // Broadphase pair buffer (flat: [i0,j0, i1,j1, ...])
  private _pairBuf: Int32Array = new Int32Array(0);
  // Broadphase pair dedup hash set (open-addressing)
  private _testedKeys: Int32Array = new Int32Array(0);
  private _testedUsed: Uint8Array = new Uint8Array(0);

  // Upload typed arrays (reused across frames, grown in ensureBuffers)
  private _bodyUpload: Float32Array | null = null;
  private _prevUpload: Float32Array | null = null;
  private _crUpload: ArrayBuffer | null = null;
  private _crUploadView: DataView | null = null;
  private _colorUpload: Uint32Array | null = null;

  // Cached bind groups (invalidated when buffers grow)
  private _primalBindGroup: GPUBindGroup | null = null;
  private _dualBindGroup: GPUBindGroup | null = null;
  private _frictionBindGroup: GPUBindGroup | null = null;

  // Persistent staging buffers for readback (grown in ensureBuffers)
  private _bodyStagingBuffer: GPUBuffer | null = null;
  private _crStagingBuffer: GPUBuffer | null = null;
  private _velRecoveryStagingBuffer: GPUBuffer | null = null;
  private _stagingMapped = false; // track mapped state for error recovery

  // Pre-allocated writeParams buffer
  private _paramsData = new ArrayBuffer(SOLVER_PARAMS_BYTES);
  private _paramsFV = new Float32Array(this._paramsData);
  private _paramsUV = new Uint32Array(this._paramsData);
  private _paramsU8 = new Uint8Array(this._paramsData);

  // Pre-allocated friction params
  private _frictionParamsU32 = new Uint32Array(1);

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
    // Prevent concurrent step() calls — async game loops (rAF at top + await step)
    // can overlap, causing staging buffer race conditions (mapped vs unmapped).
    if (this._stepping) return;
    this._stepping = true;

    try {
      await this._stepImpl();
    } finally {
      this._stepping = false;
      // If an error occurred while staging buffers were mapped, unmap them
      // so the next frame doesn't hit "buffer already mapped" errors
      if (this._stagingMapped) {
        try { this._bodyStagingBuffer?.unmap(); } catch { /* already unmapped */ }
        try { this._crStagingBuffer?.unmap(); } catch { /* already unmapped */ }
        try { this._velRecoveryStagingBuffer?.unmap(); } catch { /* already unmapped */ }
        this._stagingMapped = false;
      }
    }
  }

  private async _stepImpl(): Promise<void> {
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
    // Uses flat typed arrays to avoid per-frame object allocations
    let numPairs = 0;
    {
      const n = bodies.length;

      // Ensure flat AABB arrays are large enough
      if (this._aabbMinX.length < n) {
        const cap = Math.max(n, 64);
        this._aabbMinX = new Float64Array(cap);
        this._aabbMinY = new Float64Array(cap);
        this._aabbMinZ = new Float64Array(cap);
        this._aabbMaxX = new Float64Array(cap);
        this._aabbMaxY = new Float64Array(cap);
        this._aabbMaxZ = new Float64Array(cap);
      }
      const minX = this._aabbMinX, minY = this._aabbMinY, minZ = this._aabbMinZ;
      const maxX = this._aabbMaxX, maxY = this._aabbMaxY, maxZ = this._aabbMaxZ;

      // Compute AABBs inline into flat arrays (avoid getAABB3D object allocation)
      let totalExtent = 0, dynCount = 0;
      for (let i = 0; i < n; i++) {
        const b = bodies[i];
        if (b.colliderShape === ColliderShapeType3D.Ball) {
          const r = b.radius;
          minX[i] = b.position.x - r; maxX[i] = b.position.x + r;
          minY[i] = b.position.y - r; maxY[i] = b.position.y + r;
          minZ[i] = b.position.z - r; maxZ[i] = b.position.z + r;
        } else {
          // Cuboid: compute rotated extents inline
          const he = b.halfExtents;
          const q = b.rotation;
          // Rotation matrix columns from quaternion (inline quatRotateVec3)
          const x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
          const wx2 = q.w * x2, wy2 = q.w * y2, wz2 = q.w * z2;
          const xx2 = q.x * x2, xy2 = q.x * y2, xz2 = q.x * z2;
          const yy2 = q.y * y2, yz2 = q.y * z2, zz2 = q.z * z2;
          // Column 0 (rotated x-axis)
          const r00 = 1 - yy2 - zz2, r10 = xy2 + wz2, r20 = xz2 - wy2;
          // Column 1 (rotated y-axis)
          const r01 = xy2 - wz2, r11 = 1 - xx2 - zz2, r21 = yz2 + wx2;
          // Column 2 (rotated z-axis)
          const r02 = xz2 + wy2, r12 = yz2 - wx2, r22 = 1 - xx2 - yy2;
          const ex = Math.abs(r00) * he.x + Math.abs(r01) * he.y + Math.abs(r02) * he.z;
          const ey = Math.abs(r10) * he.x + Math.abs(r11) * he.y + Math.abs(r12) * he.z;
          const ez = Math.abs(r20) * he.x + Math.abs(r21) * he.y + Math.abs(r22) * he.z;
          minX[i] = b.position.x - ex; maxX[i] = b.position.x + ex;
          minY[i] = b.position.y - ey; maxY[i] = b.position.y + ey;
          minZ[i] = b.position.z - ez; maxZ[i] = b.position.z + ez;
        }
        if (b.type !== RigidBodyType.Fixed) {
          totalExtent += (maxX[i] - minX[i]) + (maxY[i] - minY[i]) + (maxZ[i] - minZ[i]);
          dynCount++;
        }
      }
      const cellSize = Math.max(dynCount > 0 ? (totalExtent / (dynCount * 3)) * 2 : 1, 0.5);
      const invCell = 1 / cellSize;

      // Spatial hash grid using Map (keys are cell hashes, values are body index arrays)
      // We reuse the Map across iterations but clear it each frame
      const grid = new Map<number, number[]>();

      for (let i = 0; i < n; i++) {
        const x0 = Math.floor(minX[i] * invCell), x1 = Math.floor(maxX[i] * invCell);
        const y0 = Math.floor(minY[i] * invCell), y1 = Math.floor(maxY[i] * invCell);
        const z0 = Math.floor(minZ[i] * invCell), z1 = Math.floor(maxZ[i] * invCell);
        for (let cx = x0; cx <= x1; cx++) {
          for (let cy = y0; cy <= y1; cy++) {
            for (let cz = z0; cz <= z1; cz++) {
              const k = ((cx + 0x400) * 0x100000) + ((cy + 0x400) * 0x800) + (cz + 0x400);
              let cell = grid.get(k);
              if (!cell) { cell = []; grid.set(k, cell); }
              cell.push(i);
            }
          }
        }
      }

      // Pair dedup using open-addressing hash table (avoid Set overhead)
      const maxPairsEstimate = n * 8; // reasonable upper bound
      const hashCap = maxPairsEstimate * 2; // load factor ~0.5
      if (this._testedKeys.length < hashCap) {
        this._testedKeys = new Int32Array(hashCap);
        this._testedUsed = new Uint8Array(hashCap);
      }
      const testedKeys = this._testedKeys;
      const testedUsed = this._testedUsed;
      testedUsed.fill(0); // clear

      // Ensure pair buffer is large enough
      if (this._pairBuf.length < maxPairsEstimate * 2) {
        this._pairBuf = new Int32Array(maxPairsEstimate * 2);
      }
      const pairBuf = this._pairBuf;
      numPairs = 0;
      const hasIgnorePairs = this.ignorePairs.size > 0;

      for (const cell of grid.values()) {
        const len = cell.length;
        for (let ci = 0; ci < len; ci++) {
          const i = cell[ci];
          for (let cj = ci + 1; cj < len; cj++) {
            const j = cell[cj];
            // Pair key for dedup (canonical order)
            const lo = i < j ? i : j;
            const hi = i < j ? j : i;
            const pk = lo * 65537 + hi; // unique for lo < hi < 65536

            // Open-addressing probe
            let slot = ((pk * 2654435761) >>> 0) % hashCap;
            let found = false;
            while (testedUsed[slot]) {
              if (testedKeys[slot] === pk) { found = true; break; }
              slot = (slot + 1) % hashCap;
            }
            if (found) continue;
            testedKeys[slot] = pk;
            testedUsed[slot] = 1;

            if (bodies[i].type === RigidBodyType.Fixed && bodies[j].type === RigidBodyType.Fixed) continue;
            if (hasIgnorePairs) {
              const key = `${i}-${j}`;
              if (this.ignorePairs.has(key)) continue;
            }
            // Inline AABB overlap check using flat arrays
            if (minX[i] > maxX[j] || maxX[i] < minX[j] ||
                minY[i] > maxY[j] || maxY[i] < minY[j] ||
                minZ[i] > maxZ[j] || maxZ[i] < minZ[j]) continue;

            // Grow pair buffer if needed
            if (numPairs * 2 >= pairBuf.length) {
              const newBuf = new Int32Array(pairBuf.length * 2);
              newBuf.set(pairBuf);
              this._pairBuf = newBuf;
            }
            this._pairBuf[numPairs * 2] = i;
            this._pairBuf[numPairs * 2 + 1] = j;
            numPairs++;
          }
        }
      }
    }

    const tBroadphase = performance.now();

    // Narrowphase: GJK collision detection on candidate pairs
    const pairBuf = this._pairBuf;
    for (let pi = 0; pi < numPairs; pi++) {
      const i = pairBuf[pi * 2], j = pairBuf[pi * 2 + 1];
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

    // Initialize bodies (in-place mutation to avoid object allocation)
    const gravMag = vec3Length(gravity);
    // Pre-compute gravity direction once (hoisted from per-body loop)
    const gravDirX = gravMag > 0 ? gravity.x / gravMag : 0;
    const gravDirY = gravMag > 0 ? gravity.y / gravMag : 0;
    const gravDirZ = gravMag > 0 ? gravity.z / gravMag : 0;
    // Pre-compute angular damping factor (constant for all bodies)
    const angDampFactor = 1 / (1 + 0.05 * dt);
    const dt2 = dt * dt;

    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;
      // Save prev state in-place (avoid { ...body.position } spread allocation)
      body.prevPosition.x = body.position.x;
      body.prevPosition.y = body.position.y;
      body.prevPosition.z = body.position.z;
      body.prevRotation.w = body.rotation.w;
      body.prevRotation.x = body.rotation.x;
      body.prevRotation.y = body.rotation.y;
      body.prevRotation.z = body.rotation.z;

      // Adaptive gravity weighting
      let gravWeight = 1;
      if (gravMag > 0) {
        const vx = body.velocity.x, vy = body.velocity.y, vz = body.velocity.z;
        const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (speed < 0.5) {
          const dvx = vx - body.prevVelocity.x;
          const dvy = vy - body.prevVelocity.y;
          const dvz = vz - body.prevVelocity.z;
          const dvMag = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
          if (dvMag > 0.01) {
            const accelInGravDir = (dvx * gravDirX + dvy * gravDirY + dvz * gravDirZ) / dt;
            gravWeight = Math.max(0, Math.min(1, accelInGravDir / gravMag));
          }
        }
      }

      // Implicit angular damping (in-place)
      body.angularVelocity.x *= angDampFactor;
      body.angularVelocity.y *= angDampFactor;
      body.angularVelocity.z *= angDampFactor;

      // Inertial target uses FULL gravity (the optimization objective target).
      // In-place mutation to avoid object allocation.
      const gsFullDt2 = body.gravityScale * dt2;
      body.inertialPosition.x = body.prevPosition.x + body.velocity.x * dt + gravity.x * gsFullDt2;
      body.inertialPosition.y = body.prevPosition.y + body.velocity.y * dt + gravity.y * gsFullDt2;
      body.inertialPosition.z = body.prevPosition.z + body.velocity.z * dt + gravity.z * gsFullDt2;

      // Angular integration (inline to avoid allocations)
      const awx = body.angularVelocity.x, awy = body.angularVelocity.y, awz = body.angularVelocity.z;
      const wLen = Math.sqrt(awx * awx + awy * awy + awz * awz);
      if (wLen > 1e-10) {
        const invWLen = 1 / wLen;
        const halfAngle = wLen * dt * 0.5;
        const s = Math.sin(halfAngle) * invWLen;
        const dqw = Math.cos(halfAngle);
        const dqx = awx * s, dqy = awy * s, dqz = awz * s;
        // Quaternion multiply dq * prevRotation (inline)
        const rw = body.prevRotation.w, rx = body.prevRotation.x, ry = body.prevRotation.y, rz = body.prevRotation.z;
        const nw = dqw * rw - dqx * rx - dqy * ry - dqz * rz;
        const nx = dqw * rx + dqx * rw + dqy * rz - dqz * ry;
        const ny = dqw * ry - dqx * rz + dqy * rw + dqz * rx;
        const nz = dqw * rz + dqx * ry - dqy * rx + dqz * rw;
        // Normalize
        const qLen = Math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
        const invQLen = 1 / qLen;
        body.inertialRotation.w = nw * invQLen;
        body.inertialRotation.x = nx * invQLen;
        body.inertialRotation.y = ny * invQLen;
        body.inertialRotation.z = nz * invQLen;
        // Move body to predicted position (initial guess for solver, adaptive gravity)
        body.rotation.w = nw * invQLen;
        body.rotation.x = nx * invQLen;
        body.rotation.y = ny * invQLen;
        body.rotation.z = nz * invQLen;
      } else {
        body.inertialRotation.w = body.prevRotation.w;
        body.inertialRotation.x = body.prevRotation.x;
        body.inertialRotation.y = body.prevRotation.y;
        body.inertialRotation.z = body.prevRotation.z;
        body.rotation.w = body.prevRotation.w;
        body.rotation.x = body.prevRotation.x;
        body.rotation.y = body.prevRotation.y;
        body.rotation.z = body.prevRotation.z;
      }
      // Move body to predicted position (initial guess, adaptive gravity weight)
      const gsAdaptDt2 = body.gravityScale * gravWeight * dt2;
      body.position.x = body.prevPosition.x + body.velocity.x * dt + gravity.x * gsAdaptDt2;
      body.position.y = body.prevPosition.y + body.velocity.y * dt + gravity.y * gsAdaptDt2;
      body.position.z = body.prevPosition.z + body.velocity.z * dt + gravity.z * gsAdaptDt2;
      // Save prev velocity in-place
      body.prevVelocity.x = body.velocity.x;
      body.prevVelocity.y = body.velocity.y;
      body.prevVelocity.z = body.velocity.z;
    }

    const tInit = performance.now();

    // ─── 3. CPU→GPU: Upload Buffers ───────────────────────────────
    const numBodies = bodies.length;
    const numConstraints = this.constraintRows.length;

    // Build per-body constraint indirection (indices into original constraint array)
    const { bodyRanges, constraintIndices } = this.buildConstraintIndirection(numBodies, numConstraints);

    // Ensure GPU buffers are allocated BEFORE uploading data
    this.ensureBuffers(numBodies, Math.max(numConstraints, 1), constraintIndices.length);

    // Upload body state (20 floats per body) — reuse pre-allocated buffer
    const bodyData = this._bodyUpload!;
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

    // Upload prev/inertial state (14 floats per body) — reuse pre-allocated buffer
    const prevData = this._prevUpload!;
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

    // Upload constraints in original order — reuse pre-allocated buffer
    if (numConstraints > 0) {
      const crData = this._crUpload!;
      const crView = this._crUploadView!;
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
      gpuWrite(device.queue, this.constraintBuffer, 0, new Uint8Array(crData, 0, numConstraints * CONSTRAINT_STRIDE * 4));
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

    // Use persistent velRecovery staging buffer (allocated in ensureBuffers)
    const bodyReadbackSize = numBodies * BODY_STRIDE * 4;
    const velRecoveryBuffer = config.postStabilize ? this._velRecoveryStagingBuffer : null;

    // Pre-write friction params (constant across iterations) — reuse buffer
    if (numConstraints > 0) {
      this._frictionParamsU32[0] = numConstraints;
      gpuWrite(device.queue, this.frictionParamsBuffer, 0, this._frictionParamsU32);
    }

    // Ensure cached bind groups exist (invalidated when buffers grow in ensureBuffers)
    if (!this._primalBindGroup) {
      this._primalBindGroup = device.createBindGroup({
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
    }
    if (!this._dualBindGroup) {
      this._dualBindGroup = device.createBindGroup({
        layout: this.dualBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.solverParamsBuffer } },
          { binding: 1, resource: { buffer: this.bodyStateBuffer } },
          { binding: 2, resource: { buffer: this.bodyPrevBuffer } },
          { binding: 3, resource: { buffer: this.constraintBuffer } },
        ],
      });
    }
    if (!this._frictionBindGroup) {
      this._frictionBindGroup = device.createBindGroup({
        layout: this.frictionBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.frictionParamsBuffer } },
          { binding: 1, resource: { buffer: this.constraintBuffer } },
        ],
      });
    }

    // Reusable color upload buffer
    const colorUpload = this._colorUpload!;

    for (let iter = 0; iter < totalIterations; iter++) {
      const isStabilization = config.postStabilize && iter === totalIterations - 1;

      // ─── 4a. Primal update: dispatch per color group ──────────
      for (const colorGroup of this.colorGroups) {
        const groupLen = colorGroup.bodyIndices.length;
        if (groupLen === 0) continue;

        this.writeParams(dt, gravity, config, numBodies, numConstraints,
          groupLen, isStabilization);

        // Copy color indices into reusable buffer
        const indices = colorGroup.bodyIndices;
        for (let k = 0; k < groupLen; k++) colorUpload[k] = indices[k];
        gpuWrite(device.queue, this.colorIndicesBuffer, 0, colorUpload.subarray(0, groupLen));

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.primalPipeline);
        pass.setBindGroup(0, this._primalBindGroup);
        pass.dispatchWorkgroups(Math.ceil(groupLen / WORKGROUP_SIZE));
        pass.end();
        device.queue.submit([encoder.finish()]);
      }

      // ─── 4b. Dual update + friction coupling ─────────────────
      if (!isStabilization && numConstraints > 0) {
        this.writeParams(dt, gravity, config, numBodies, numConstraints, 0, false);

        const encoder = device.createCommandEncoder();

        // Dual: update all constraint lambdas
        const dualPass = encoder.beginComputePass();
        dualPass.setPipeline(this.dualPipeline);
        dualPass.setBindGroup(0, this._dualBindGroup);
        dualPass.dispatchWorkgroups(Math.ceil(numConstraints / WORKGROUP_SIZE));
        dualPass.end();

        // Friction coupling
        const numContactTriplets = Math.ceil(numConstraints / 3);
        const fricPass = encoder.beginComputePass();
        fricPass.setPipeline(this.frictionPipeline);
        fricPass.setBindGroup(0, this._frictionBindGroup);
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

    // Check for GPU validation errors — throw so callers can display them
    const validationError = await device.popErrorScope();
    if (validationError) {
      throw new Error('GPU validation error: ' + validationError.message);
    }

    const tDispatch = performance.now();

    // ─── 5. GPU→CPU: Read Back Results ────────────────────────────
    // Persistent staging buffers (allocated in ensureBuffers, reused across frames)
    const bodyStagingBuffer = this._bodyStagingBuffer!;
    const crStagingBuffer = (numConstraints > 0) ? this._crStagingBuffer : null;
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(this.bodyStateBuffer, 0, bodyStagingBuffer, 0, bodyReadbackSize);
    if (crStagingBuffer && numConstraints > 0) {
      const crReadbackSize = numConstraints * CONSTRAINT_STRIDE * 4;
      copyEncoder.copyBufferToBuffer(this.constraintBuffer, 0, crStagingBuffer, 0, crReadbackSize);
    }

    device.queue.submit([copyEncoder.finish()]);

    // Map ALL staging buffers in parallel — single GPU fence wait
    const mapPromises: Promise<void>[] = [bodyStagingBuffer.mapAsync(GPUMapMode.READ)];
    if (velRecoveryBuffer) mapPromises.push(velRecoveryBuffer.mapAsync(GPUMapMode.READ));
    if (crStagingBuffer && numConstraints > 0) mapPromises.push(crStagingBuffer.mapAsync(GPUMapMode.READ));
    await Promise.all(mapPromises);
    this._stagingMapped = true;

    // Read directly from mapped ranges (no .slice(0) copy needed — we read inline)
    const bodyResult = new Float32Array(bodyStagingBuffer.getMappedRange());

    let velRecoveryResult: Float32Array | null = null;
    if (velRecoveryBuffer) {
      velRecoveryResult = new Float32Array(velRecoveryBuffer.getMappedRange());
    }

    // ─── 6. CPU: Apply Results ────────────────────────────────────
    // Use pre-stabilization positions for velocity recovery (matching CPU solver),
    // but final (post-stabilization) positions for body state
    const velSource = velRecoveryResult ?? bodyResult;

    const MAX_LIN_VEL = 100;
    const MAX_ANG_VEL = 50;
    const invDt = 1 / dt;
    const twoInvDt = 2 * invDt;

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

      // BDF1 velocity recovery from pre-stabilization positions (in-place)
      const voff = i * BODY_STRIDE;
      let vx = (velSource[voff + 0] - body.prevPosition.x) * invDt;
      let vy = (velSource[voff + 1] - body.prevPosition.y) * invDt;
      let vz = (velSource[voff + 2] - body.prevPosition.z) * invDt;
      // Clamp recovered linear velocity
      const vLen = Math.sqrt(vx * vx + vy * vy + vz * vz);
      if (vLen > MAX_LIN_VEL) {
        const s = MAX_LIN_VEL / vLen;
        vx *= s; vy *= s; vz *= s;
      }
      body.velocity.x = vx;
      body.velocity.y = vy;
      body.velocity.z = vz;

      // Angular velocity from quaternion difference (inline, avoid object allocation)
      const vqw = velSource[voff + 3];
      const vqx = velSource[voff + 4];
      const vqy = velSource[voff + 5];
      const vqz = velSource[voff + 6];
      const prw = body.prevRotation.w, prx = -body.prevRotation.x;
      const pry = -body.prevRotation.y, prz = -body.prevRotation.z;
      // Inline quatMul
      const dqx = vqw * prx + vqx * prw + vqy * prz - vqz * pry;
      const dqy = vqw * pry - vqx * prz + vqy * prw + vqz * prx;
      const dqz = vqw * prz + vqx * pry - vqy * prx + vqz * prw;
      let wx = dqx * twoInvDt, wy = dqy * twoInvDt, wz = dqz * twoInvDt;
      // Clamp recovered angular velocity
      const wLen = Math.sqrt(wx * wx + wy * wy + wz * wz);
      if (wLen > MAX_ANG_VEL) {
        const s = MAX_ANG_VEL / wLen;
        wx *= s; wy *= s; wz *= s;
      }
      body.angularVelocity.x = wx;
      body.angularVelocity.y = wy;
      body.angularVelocity.z = wz;
    }

    // Readback constraint lambdas for warmstarting
    if (crStagingBuffer && numConstraints > 0) {
      const crResult = new DataView(crStagingBuffer.getMappedRange());
      for (let i = 0; i < numConstraints; i++) {
        const byteOff = i * CONSTRAINT_STRIDE * 4;
        this.constraintRows[i].lambda = crResult.getFloat32(byteOff + 120, true);
        this.constraintRows[i].penalty = crResult.getFloat32(byteOff + 124, true);
      }
    }

    // Unmap all persistent staging buffers (must happen after all reads)
    bodyStagingBuffer.unmap();
    if (crStagingBuffer && numConstraints > 0) crStagingBuffer.unmap();
    if (velRecoveryBuffer) velRecoveryBuffer.unmap();
    this._stagingMapped = false;

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

  // Reusable buffers for buildConstraintIndirection (avoid per-frame allocation)
  private _ciCounts: Uint32Array = new Uint32Array(0);
  private _ciBodyRanges: Uint32Array = new Uint32Array(0);
  private _ciAllIndices: Uint32Array = new Uint32Array(0);

  /** Build per-body constraint index lists for indirection on GPU (two-pass, zero intermediate allocation) */
  private buildConstraintIndirection(numBodies: number, numConstraints: number): {
    bodyRanges: Uint32Array;
    constraintIndices: Uint32Array;
  } {
    const rows = this.constraintRows;

    // Ensure count buffer is large enough
    if (this._ciCounts.length < numBodies) {
      this._ciCounts = new Uint32Array(Math.max(numBodies, 64));
    }
    if (this._ciBodyRanges.length < numBodies * 2) {
      this._ciBodyRanges = new Uint32Array(Math.max(numBodies * 2, 128));
    }
    const counts = this._ciCounts;
    counts.fill(0, 0, numBodies);

    // Pass 1: count constraints per body
    for (let ci = 0; ci < numConstraints; ci++) {
      const row = rows[ci];
      if (!row.active) continue;
      if (row.bodyA >= 0 && row.bodyA < numBodies) counts[row.bodyA]++;
      if (row.bodyB >= 0 && row.bodyB < numBodies) counts[row.bodyB]++;
    }

    // Compute offsets (prefix sum) and build bodyRanges
    const bodyRanges = this._ciBodyRanges;
    let totalIndices = 0;
    for (let i = 0; i < numBodies; i++) {
      bodyRanges[i * 2 + 0] = totalIndices;
      bodyRanges[i * 2 + 1] = counts[i];
      totalIndices += counts[i];
    }

    // Ensure all-indices buffer is large enough
    if (this._ciAllIndices.length < totalIndices) {
      this._ciAllIndices = new Uint32Array(Math.max(totalIndices, 64));
    }
    const allIndices = this._ciAllIndices;

    // Pass 2: fill indices (use counts as write cursors, decrement from offset)
    // Reset counts to use as fill pointers
    counts.fill(0, 0, numBodies);
    for (let ci = 0; ci < numConstraints; ci++) {
      const row = rows[ci];
      if (!row.active) continue;
      if (row.bodyA >= 0 && row.bodyA < numBodies) {
        allIndices[bodyRanges[row.bodyA * 2] + counts[row.bodyA]++] = ci;
      }
      if (row.bodyB >= 0 && row.bodyB < numBodies) {
        allIndices[bodyRanges[row.bodyB * 2] + counts[row.bodyB]++] = ci;
      }
    }

    return {
      bodyRanges: bodyRanges.subarray(0, numBodies * 2),
      constraintIndices: allIndices.subarray(0, totalIndices),
    };
  }

  /** Write solver params uniform (3D: 12 fields) — reuses pre-allocated buffer */
  private writeParams(
    dt: number, gravity: Vec3, config: SolverConfig,
    numBodies: number, numConstraints: number,
    numBodiesInGroup: number, isStabilization: boolean,
  ): void {
    const fv = this._paramsFV;
    const uv = this._paramsUV;
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
    gpuWrite(this.gpu.device.queue, this.solverParamsBuffer, 0, this._paramsU8);
  }

  private maxConstraintIndices = 0;

  /** Ensure GPU buffers and reusable typed arrays are large enough */
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

      // Reusable upload typed arrays
      this._bodyUpload = new Float32Array(this.maxBodies * BODY_STRIDE);
      this._prevUpload = new Float32Array(this.maxBodies * BODY_PREV_STRIDE);
      this._colorUpload = new Uint32Array(this.maxBodies);

      // Persistent staging buffers for readback (body state + velocity recovery snapshot)
      const MAP_READ_DST = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
      const bodyStagingSize = this.maxBodies * BODY_STRIDE * 4;
      if (this._bodyStagingBuffer) this._bodyStagingBuffer.destroy();
      this._bodyStagingBuffer = device.createBuffer({ size: bodyStagingSize, usage: MAP_READ_DST });
      if (this._velRecoveryStagingBuffer) this._velRecoveryStagingBuffer.destroy();
      this._velRecoveryStagingBuffer = device.createBuffer({ size: bodyStagingSize, usage: MAP_READ_DST });

      // Invalidate cached bind groups (buffers changed)
      this._primalBindGroup = null;
      this._dualBindGroup = null;
      this._frictionBindGroup = null;
    }

    if (numConstraints > this.maxConstraints) {
      this.maxConstraints = Math.max(numConstraints, this.maxConstraints * 2, 64);
      if (this.constraintBuffer) this.constraintBuffer.destroy();
      this.constraintBuffer = device.createBuffer({ size: this.maxConstraints * CONSTRAINT_STRIDE * 4, usage: STORAGE });

      // Reusable constraint upload buffer
      this._crUpload = new ArrayBuffer(this.maxConstraints * CONSTRAINT_STRIDE * 4);
      this._crUploadView = new DataView(this._crUpload);

      // Persistent constraint staging buffer for readback
      if (this._crStagingBuffer) this._crStagingBuffer.destroy();
      this._crStagingBuffer = device.createBuffer({
        size: this.maxConstraints * CONSTRAINT_STRIDE * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      // Invalidate cached bind groups (constraint buffer changed)
      this._primalBindGroup = null;
      this._dualBindGroup = null;
      this._frictionBindGroup = null;
    }

    const ciCount = Math.max(numConstraintIndices, 1);
    if (ciCount > this.maxConstraintIndices) {
      this.maxConstraintIndices = Math.max(ciCount, this.maxConstraintIndices * 2, 64);
      if (this.constraintIndicesBuffer) this.constraintIndicesBuffer.destroy();
      this.constraintIndicesBuffer = device.createBuffer({ size: this.maxConstraintIndices * 4, usage: STORAGE });

      // Invalidate primal bind group (constraint indices buffer changed)
      this._primalBindGroup = null;
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
      const lo = row.bodyA < row.bodyB ? row.bodyA : row.bodyB;
      const hi = row.bodyA < row.bodyB ? row.bodyB : row.bodyA;
      const key = lo * 65536 + hi;
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
      const lo = row.bodyA < row.bodyB ? row.bodyA : row.bodyB;
      const hi = row.bodyA < row.bodyB ? row.bodyB : row.bodyA;
      const key = lo * 65536 + hi;
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
      this.bodyConstraintRangesBuffer, this.constraintIndicesBuffer,
      this._bodyStagingBuffer, this._crStagingBuffer, this._velRecoveryStagingBuffer]) {
      if (buf) buf.destroy();
    }
    this._bodyStagingBuffer = null;
    this._crStagingBuffer = null;
    this._velRecoveryStagingBuffer = null;
  }
}
