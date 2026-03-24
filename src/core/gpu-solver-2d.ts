/**
 * GPU-accelerated AVBD 2D solver.
 *
 * Dispatches WGSL compute shaders via WebGPU for the primal and dual updates.
 * This is the primary solver path — the CPU solver exists as a fallback.
 *
 * Data flow per step:
 * 1. CPU: broadphase + narrowphase collision detection (hard to parallelize)
 * 2. CPU: graph coloring for conflict-free parallel dispatch
 * 3. CPU→GPU: upload body state, prev state, sorted constraints, color groups
 * 4. GPU: for each solver iteration:
 *    a. For each color group: dispatch primal update shader (one pass per color)
 *    b. Dispatch dual update shader (all constraints in one pass)
 * 5. GPU→CPU: async readback of body positions + constraint lambdas
 * 6. CPU: velocity recovery, contact caching
 */

import type { Vec2, SolverConfig, ColorGroup, StepTimings } from './types.js';
import { RigidBodyType, DEFAULT_SOLVER_CONFIG_2D } from './types.js';
import { ForceType } from './types.js';
import type { Body2D } from './rigid-body.js';
import { BodyStore2D } from './rigid-body.js';
import type { ConstraintRow } from '../constraints/constraint.js';
import { ConstraintStore } from '../constraints/constraint.js';
import { createContactConstraintRows } from '../constraints/contact.js';
import { collide2D } from '../2d/collision-sat.js';
import { aabb2DOverlap, vec2Scale, vec2Length } from './math.js';
import { computeGraphColoring } from './graph-coloring.js';
import { GPUContext } from './gpu-context.js';
import { PRIMAL_UPDATE_2D_WGSL, DUAL_UPDATE_WGSL, FRICTION_COUPLING_WGSL } from '../shaders/embedded.js';

// ─── GPU Buffer Layout Constants ────────────────────────────────────────────

/** Body state: [x, y, angle, vx, vy, omega, mass, inertia] = 8 floats */
const BODY_STRIDE = 8;
/** Body prev: [prev_x, prev_y, prev_angle, inertial_x, inertial_y, inertial_angle, 0, 0] = 8 floats */
const BODY_PREV_STRIDE = 8;
/**
 * Constraint row: matches WGSL ConstraintRow struct exactly = 28 floats (112 bytes)
 * [body_a(i32), body_b(i32), force_type(u32), _pad(u32),   // 4 fields = 16 bytes
 *  jacobian_a(vec4), jacobian_b(vec4),                       // 2 vec4  = 32 bytes
 *  hessian_diag_a(vec4), hessian_diag_b(vec4),              // 2 vec4  = 32 bytes
 *  c, c0, lambda, penalty, stiffness, fmin, fmax, active]   // 8 fields = 32 bytes
 * Total: 112 bytes = 28 floats
 */
const CONSTRAINT_STRIDE = 28;
/**
 * SolverParams uniform: matches WGSL struct = 10 fields = 40 bytes
 * [dt, gravity_x, gravity_y, penalty_min, penalty_max, beta,
 *  num_bodies(u32), num_constraints(u32), num_bodies_in_group(u32), is_stabilization(u32)]
 */
const SOLVER_PARAMS_FLOATS = 10;
const SOLVER_PARAMS_BYTES = SOLVER_PARAMS_FLOATS * 4;
const WORKGROUP_SIZE = 64;

/** Helper to write typed arrays to GPU buffers (works around strict @webgpu/types) */
function gpuWrite(queue: GPUQueue, buffer: GPUBuffer, offset: number, data: { buffer: ArrayBufferLike; byteOffset: number; byteLength: number }): void {
  (queue as any).writeBuffer(buffer, offset, data);
}

// ─── GPU Solver ─────────────────────────────────────────────────────────────

export class GPUSolver2D {
  config: SolverConfig;
  bodyStore: BodyStore2D;
  constraintStore: ConstraintStore;
  ignorePairs: Set<string> = new Set();
  jointConstraintIndices: number[] = [];
  colorGroups: ColorGroup[] = [];
  lastTimings: StepTimings | null = null;

  private gpu: GPUContext;
  private initialized = false;
  private _stepping = false;

  // GPU resources
  private bodyStateBuffer!: GPUBuffer;
  private bodyPrevBuffer!: GPUBuffer;
  private constraintBuffer!: GPUBuffer;
  private solverParamsBuffer!: GPUBuffer;
  private colorIndicesBuffer!: GPUBuffer;
  private bodyConstraintRangesBuffer!: GPUBuffer;
  private constraintIndicesBuffer!: GPUBuffer;

  // Persistent staging buffers for readback (grown in ensureBuffers)
  private _bodyStagingBuffer: GPUBuffer | null = null;
  private _crStagingBuffer: GPUBuffer | null = null;
  private _velRecoveryStagingBuffer: GPUBuffer | null = null;
  private _stagingMapped = false;

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
    this.config = { ...DEFAULT_SOLVER_CONFIG_2D, ...config };
    this.bodyStore = new BodyStore2D();
    this.constraintStore = new ConstraintStore();
  }

  /** Initialize GPU pipelines. Called automatically on first step. */
  async init(): Promise<void> {
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

    const primalModule = device.createShaderModule({ code: PRIMAL_UPDATE_2D_WGSL });
    const primalInfo = await primalModule.getCompilationInfo();
    for (const msg of primalInfo.messages) {
      if (msg.type === 'error') {
        throw new Error(`Primal shader error at line ${msg.lineNum}: ${msg.message}`);
      }
    }
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

    const dualModule = device.createShaderModule({ code: DUAL_UPDATE_WGSL });
    const dualInfo = await dualModule.getCompilationInfo();
    for (const msg of dualInfo.messages) {
      if (msg.type === 'error') {
        throw new Error(`Dual shader error at line ${msg.lineNum}: ${msg.message}`);
      }
    }
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

    const frictionModule = device.createShaderModule({ code: FRICTION_COUPLING_WGSL });
    this.frictionPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frictionBindGroupLayout] }),
      compute: { module: frictionModule, entryPoint: 'main' },
    });

    // Solver params uniform (fixed size)
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
    if (!this.initialized) await this.init();
    // Prevent concurrent step() calls — with async GPU readback, rAF at top of loop
    // can overlap, causing staging buffer race conditions.
    if (this._stepping) return;
    this._stepping = true;
    try {
      await this._stepImpl();
    } finally {
      this._stepping = false;
      if (this._stagingMapped) {
        try { this._bodyStagingBuffer?.unmap(); } catch { /* already unmapped */ }
        try { this._crStagingBuffer?.unmap(); } catch { /* already unmapped */ }
        try { this._velRecoveryStagingBuffer?.unmap(); } catch { /* already unmapped */ }
        this._stagingMapped = false;
      }
    }
  }

  private async _stepImpl(): Promise<void> {
    const t0 = performance.now();
    const { config, bodyStore, constraintStore, gpu } = this;
    const dt = config.dt;
    const gravity = config.gravity as Vec2;
    const bodies = bodyStore.bodies;
    const device = gpu.device;

    if (bodies.length === 0) return;

    // ─── 1. CPU: Collision Detection (spatial hash broadphase) ─────
    constraintStore.clearContacts();
    const gpu2dCandidatePairs: [number, number][] = [];

    {
      const n = bodies.length;
      const aabbs = new Array(n);
      for (let i = 0; i < n; i++) aabbs[i] = bodyStore.getAABB(bodies[i]);

      let totalExtent = 0, dynCount = 0;
      for (let i = 0; i < n; i++) {
        if (bodies[i].type === RigidBodyType.Dynamic) {
          totalExtent += (aabbs[i].maxX - aabbs[i].minX) + (aabbs[i].maxY - aabbs[i].minY);
          dynCount++;
        }
      }
      const cellSize = Math.max(dynCount > 0 ? (totalExtent / (dynCount * 2)) * 2 : 1, 0.5);
      const invCell = 1 / cellSize;

      const grid = new Map<number, number[]>();
      for (let i = 0; i < n; i++) {
        const a = aabbs[i];
        const x0 = Math.floor(a.minX * invCell), x1 = Math.floor(a.maxX * invCell);
        const y0 = Math.floor(a.minY * invCell), y1 = Math.floor(a.maxY * invCell);
        for (let cx = x0; cx <= x1; cx++) {
          for (let cy = y0; cy <= y1; cy++) {
            const k = (cx + 0x8000) * 0x10000 + (cy + 0x8000);
            let cell = grid.get(k);
            if (!cell) { cell = []; grid.set(k, cell); }
            cell.push(i);
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
            const a = bodies[i], b = bodies[j];
            if (a.type !== RigidBodyType.Dynamic && b.type !== RigidBodyType.Dynamic) continue;
            const key = i < j ? `${i}-${j}` : `${j}-${i}`;
            if (this.ignorePairs.has(key)) continue;
            if (!aabb2DOverlap(aabbs[i], aabbs[j])) continue;
            gpu2dCandidatePairs.push([i, j]);
          }
        }
      }
    }

    const tBroadphase = performance.now();

    // Narrowphase: SAT collision detection on candidate pairs
    for (const [i, j] of gpu2dCandidatePairs) {
      const manifold = collide2D(bodies[i], bodies[j]);
      if (manifold) {
        const rows = createContactConstraintRows(manifold, bodies[i], bodies[j], config.penaltyMin, Infinity, config.dt);
        constraintStore.addRows(rows);
      }
    }
    constraintStore.warmstartContacts();

    const tCollision = performance.now();

    // ─── 2. CPU: Warmstart & Initialize ───────────────────────────
    for (const row of constraintStore.rows) {
      if (!row.active) continue;
      row.penalty *= config.gamma;
      row.penalty = Math.max(config.penaltyMin, Math.min(config.penaltyMax, row.penalty));
      if (row.penalty > row.stiffness) row.penalty = row.stiffness;
      if (!config.postStabilize) row.lambda *= config.alpha * config.gamma;
    }

    // Graph coloring
    const constraintPairs = constraintStore.getConstraintPairs();
    const fixedBodies = new Set<number>();
    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) fixedBodies.add(body.index);
    }
    this.colorGroups = computeGraphColoring(bodies.length, constraintPairs, fixedBodies);

    // Initialize bodies
    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;
      body.angularVelocity = Math.max(-50, Math.min(50, body.angularVelocity));
      body.prevPosition = { ...body.position };
      body.prevAngle = body.angle;

      // Adaptive gravity weighting: only for slow-moving bodies to avoid artificial bounce
      const gravMag = vec2Length(gravity);
      let gravWeight = 1;
      if (gravMag > 0) {
        const speed = Math.sqrt(body.velocity.x * body.velocity.x + body.velocity.y * body.velocity.y);
        if (speed < 0.5) {
          const dvx = body.velocity.x - body.prevVelocity.x;
          const dvy = body.velocity.y - body.prevVelocity.y;
          const dvMag = Math.sqrt(dvx * dvx + dvy * dvy);
          if (dvMag > 0.01) {
            const gravDir = { x: gravity.x / gravMag, y: gravity.y / gravMag };
            const accelInGravDir = (dvx * gravDir.x + dvy * gravDir.y) / dt;
            gravWeight = Math.max(0, Math.min(1, accelInGravDir / gravMag));
          }
        }
      }

      let vx = body.velocity.x, vy = body.velocity.y, omega = body.angularVelocity;
      if (body.linearDamping > 0) { const f = 1 / (1 + body.linearDamping * dt); vx *= f; vy *= f; }
      { const totalAngDamp = body.angularDamping + 0.05; omega *= 1 / (1 + totalAngDamp * dt); }

      body.inertialPosition = {
        x: body.position.x + vx * dt + gravity.x * body.gravityScale * gravWeight * dt * dt,
        y: body.position.y + vy * dt + gravity.y * body.gravityScale * gravWeight * dt * dt,
      };
      body.inertialAngle = body.angle + omega * dt;
      body.prevVelocity = { ...body.velocity };
    }

    const tInit = performance.now();

    // ─── 3. CPU→GPU: Upload Buffers ───────────────────────────────
    const numBodies = bodies.length;
    const numConstraints = constraintStore.rows.length;

    // Build per-body constraint indirection (indices into original constraint array)
    const { bodyRanges, constraintIndices } = this.buildConstraintIndirection(numBodies, numConstraints);

    this.ensureBuffers(numBodies, Math.max(numConstraints, 1), constraintIndices.length);

    // Upload body state
    const bodyData = new Float32Array(numBodies * BODY_STRIDE);
    for (let i = 0; i < numBodies; i++) {
      const b = bodies[i];
      const off = i * BODY_STRIDE;
      bodyData[off + 0] = b.position.x;
      bodyData[off + 1] = b.position.y;
      bodyData[off + 2] = b.angle;
      bodyData[off + 3] = b.velocity.x;
      bodyData[off + 4] = b.velocity.y;
      bodyData[off + 5] = b.angularVelocity;
      bodyData[off + 6] = b.mass;
      bodyData[off + 7] = b.inertia;
    }
    gpuWrite(device.queue, this.bodyStateBuffer, 0, bodyData);

    // Upload prev/inertial state
    const prevData = new Float32Array(numBodies * BODY_PREV_STRIDE);
    for (let i = 0; i < numBodies; i++) {
      const b = bodies[i];
      const off = i * BODY_PREV_STRIDE;
      prevData[off + 0] = b.prevPosition.x;
      prevData[off + 1] = b.prevPosition.y;
      prevData[off + 2] = b.prevAngle;
      prevData[off + 3] = b.inertialPosition.x;
      prevData[off + 4] = b.inertialPosition.y;
      prevData[off + 5] = b.inertialAngle;
      prevData[off + 6] = 0;
      prevData[off + 7] = 0;
    }
    gpuWrite(device.queue, this.bodyPrevBuffer, 0, prevData);

    const rows = this.constraintStore.rows;

    // Upload constraints in original order (28 floats = 112 bytes per row)
    if (numConstraints > 0) {
      const crData = new ArrayBuffer(numConstraints * CONSTRAINT_STRIDE * 4);
      const crView = new DataView(crData);
      for (let i = 0; i < numConstraints; i++) {
        const row = rows[i];
        const byteOff = i * CONSTRAINT_STRIDE * 4;
        crView.setInt32(byteOff + 0, row.bodyA, true);
        crView.setInt32(byteOff + 4, row.bodyB, true);
        crView.setUint32(byteOff + 8, row.type, true);
        crView.setUint32(byteOff + 12, 0, true); // padding
        // jacobian_a (vec4)
        crView.setFloat32(byteOff + 16, row.jacobianA[0], true);
        crView.setFloat32(byteOff + 20, row.jacobianA[1], true);
        crView.setFloat32(byteOff + 24, row.jacobianA[2], true);
        crView.setFloat32(byteOff + 28, 0, true);
        // jacobian_b (vec4)
        crView.setFloat32(byteOff + 32, row.jacobianB[0], true);
        crView.setFloat32(byteOff + 36, row.jacobianB[1], true);
        crView.setFloat32(byteOff + 40, row.jacobianB[2], true);
        crView.setFloat32(byteOff + 44, 0, true);
        // hessian_diag_a (vec4)
        crView.setFloat32(byteOff + 48, row.hessianDiagA[0], true);
        crView.setFloat32(byteOff + 52, row.hessianDiagA[1], true);
        crView.setFloat32(byteOff + 56, row.hessianDiagA[2], true);
        crView.setFloat32(byteOff + 60, 0, true);
        // hessian_diag_b (vec4) — .w stores friction coefficient mu for contact rows
        crView.setFloat32(byteOff + 64, row.hessianDiagB[0], true);
        crView.setFloat32(byteOff + 68, row.hessianDiagB[1], true);
        crView.setFloat32(byteOff + 72, row.hessianDiagB[2], true);
        // Pack mu (friction coefficient) into hessian_diag_b.w for GPU dual shader
        let mu = 0.5;
        if (row.type === ForceType.Contact && row.bodyA >= 0 && row.bodyB >= 0) {
          const bA = this.bodyStore.bodies[row.bodyA];
          const bB = this.bodyStore.bodies[row.bodyB];
          if (bA && bB) mu = Math.sqrt(bA.friction * bB.friction);
        }
        crView.setFloat32(byteOff + 76, mu, true);
        // scalar fields (8 fields)
        crView.setFloat32(byteOff + 80, row.c, true);
        crView.setFloat32(byteOff + 84, row.c0, true);
        crView.setFloat32(byteOff + 88, row.lambda, true);
        crView.setFloat32(byteOff + 92, row.penalty, true);
        crView.setFloat32(byteOff + 96, isFinite(row.stiffness) ? row.stiffness : 1e30, true);
        crView.setFloat32(byteOff + 100, isFinite(row.fmin) ? row.fmin : -1e30, true);
        crView.setFloat32(byteOff + 104, isFinite(row.fmax) ? row.fmax : 1e30, true);
        crView.setUint32(byteOff + 108, row.active ? 1 : 0, true);
      }
      gpuWrite(device.queue, this.constraintBuffer, 0, new Uint8Array(crData));
    }

    // Upload body constraint ranges and indirection indices
    gpuWrite(device.queue, this.bodyConstraintRangesBuffer, 0, bodyRanges);
    if (constraintIndices.length > 0) {
      gpuWrite(device.queue, this.constraintIndicesBuffer, 0, constraintIndices);
    }

    const tUpload = performance.now();

    // ─── 4. GPU: Solver Iterations ────────────────────────────────
    const totalIterations = config.postStabilize ? config.iterations + 1 : config.iterations;
    const velocityRecoveryIter = config.iterations - 1;

    // Push error scope to catch GPU validation errors
    device.pushErrorScope('validation');

    // Use persistent velRecovery staging buffer (allocated in ensureBuffers)
    const bodyReadbackSize = numBodies * BODY_STRIDE * 4;
    const velRecoveryBuffer = config.postStabilize ? this._velRecoveryStagingBuffer : null;

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
        // Per AVBD reference (manifold.cpp: computeConstraint).
        const numContactPairs = Math.ceil(numConstraints / 2);
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
        fricPass.dispatchWorkgroups(Math.ceil(numContactPairs / WORKGROUP_SIZE));
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

    // Read directly from mapped ranges (no .slice(0) copy — we read inline before unmap)
    const bodyResult = new Float32Array(bodyStagingBuffer.getMappedRange());

    let velRecoveryResult: Float32Array | null = null;
    if (velRecoveryBuffer) {
      velRecoveryResult = new Float32Array(velRecoveryBuffer.getMappedRange());
    }

    // ─── 6. CPU: Apply Results ────────────────────────────────────
    // Use pre-stabilization positions for velocity recovery (matching CPU solver),
    // but final (post-stabilization) positions for body state
    const velSource = velRecoveryResult ?? bodyResult;

    const MAX_LINEAR_VELOCITY = 100;
    const MAX_ANGULAR_VELOCITY = 50;
    const invDt = 1 / dt;

    for (let i = 0; i < numBodies; i++) {
      const body = bodies[i];
      if (body.type !== RigidBodyType.Dynamic) continue;

      const off = i * BODY_STRIDE;
      body.position.x = bodyResult[off + 0];
      body.position.y = bodyResult[off + 1];
      body.angle = bodyResult[off + 2];

      // BDF1 velocity recovery from pre-stabilization positions (no object allocation)
      let vx = (velSource[off + 0] - body.prevPosition.x) * invDt;
      let vy = (velSource[off + 1] - body.prevPosition.y) * invDt;
      const vMag = Math.sqrt(vx * vx + vy * vy);
      if (vMag > MAX_LINEAR_VELOCITY) {
        const scale = MAX_LINEAR_VELOCITY / vMag;
        vx *= scale;
        vy *= scale;
      }
      body.velocity.x = vx;
      body.velocity.y = vy;

      let av = (velSource[off + 2] - body.prevAngle) * invDt;
      if (av > MAX_ANGULAR_VELOCITY) av = MAX_ANGULAR_VELOCITY;
      else if (av < -MAX_ANGULAR_VELOCITY) av = -MAX_ANGULAR_VELOCITY;
      body.angularVelocity = av;
    }

    // Readback constraint lambdas for warmstarting
    if (crStagingBuffer && numConstraints > 0) {
      const crResult = new DataView(crStagingBuffer.getMappedRange());
      for (let i = 0; i < numConstraints; i++) {
        const byteOff = i * CONSTRAINT_STRIDE * 4;
        rows[i].lambda = crResult.getFloat32(byteOff + 88, true);
        rows[i].penalty = crResult.getFloat32(byteOff + 92, true);
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
      numConstraints: constraintStore.rows.length,
    };
  }

  /** Build per-body constraint index lists for indirection on GPU */
  private buildConstraintIndirection(numBodies: number, numConstraints: number): {
    bodyRanges: Uint32Array;
    constraintIndices: Uint32Array;
  } {
    const rows = this.constraintStore.rows;

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

  /** Write solver params uniform */
  private writeParams(
    dt: number, gravity: Vec2, config: SolverConfig,
    numBodies: number, numConstraints: number,
    numBodiesInGroup: number, isStabilization: boolean,
  ): void {
    const data = new ArrayBuffer(SOLVER_PARAMS_BYTES);
    const fv = new Float32Array(data);
    const uv = new Uint32Array(data);
    fv[0] = dt;
    fv[1] = gravity.x;
    fv[2] = gravity.y;
    fv[3] = config.penaltyMin;
    fv[4] = config.penaltyMax;
    fv[5] = config.beta;
    uv[6] = numBodies;
    uv[7] = numConstraints;
    uv[8] = numBodiesInGroup;
    uv[9] = isStabilization ? 1 : 0;
    gpuWrite(this.gpu.device.queue, this.solverParamsBuffer, 0, new Uint8Array(data));
  }

  private maxConstraintIndices = 0;

  /** Ensure GPU buffers are large enough */
  private ensureBuffers(numBodies: number, numConstraints: number, numConstraintIndices: number): void {
    const device = this.gpu.device;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const MAP_READ_DST = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

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

      // Persistent staging buffers for readback (body state + velocity recovery snapshot)
      const bodyStagingSize = this.maxBodies * BODY_STRIDE * 4;
      if (this._bodyStagingBuffer) this._bodyStagingBuffer.destroy();
      this._bodyStagingBuffer = device.createBuffer({ size: bodyStagingSize, usage: MAP_READ_DST });
      if (this._velRecoveryStagingBuffer) this._velRecoveryStagingBuffer.destroy();
      this._velRecoveryStagingBuffer = device.createBuffer({ size: bodyStagingSize, usage: MAP_READ_DST });
    }

    if (numConstraints > this.maxConstraints) {
      this.maxConstraints = Math.max(numConstraints, this.maxConstraints * 2, 64);
      if (this.constraintBuffer) this.constraintBuffer.destroy();
      this.constraintBuffer = device.createBuffer({ size: this.maxConstraints * CONSTRAINT_STRIDE * 4, usage: STORAGE });

      // Persistent constraint staging buffer for readback
      if (this._crStagingBuffer) this._crStagingBuffer.destroy();
      this._crStagingBuffer = device.createBuffer({
        size: this.maxConstraints * CONSTRAINT_STRIDE * 4,
        usage: MAP_READ_DST,
      });
    }

    const ciCount = Math.max(numConstraintIndices, 1);
    if (ciCount > this.maxConstraintIndices) {
      this.maxConstraintIndices = Math.max(ciCount, this.maxConstraintIndices * 2, 64);
      if (this.constraintIndicesBuffer) this.constraintIndicesBuffer.destroy();
      this.constraintIndicesBuffer = device.createBuffer({ size: this.maxConstraintIndices * 4, usage: STORAGE });
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
