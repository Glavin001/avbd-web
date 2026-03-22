/**
 * GPU-accelerated AVBD 2D solver.
 *
 * This is the actual WebGPU execution path that dispatches WGSL compute
 * shaders to perform the primal and dual updates on the GPU.
 *
 * Data flow per step:
 * 1. CPU: broadphase + narrowphase collision detection
 * 2. CPU: graph coloring for parallel dispatch
 * 3. CPU→GPU: upload body state, prev state, constraints, color groups
 * 4. GPU: for each solver iteration:
 *    a. For each color group: dispatch primal update shader
 *    b. Dispatch dual update shader (all constraints)
 * 5. GPU→CPU: read back body positions
 * 6. CPU: velocity recovery, contact caching
 */

import type { Vec2, SolverConfig, ColorGroup } from './types.js';
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

// Shader source is provided via setShaderSource() or loaded by the bundler.
// This avoids hard dependency on fs (browser) or ?raw imports (bundler-specific).

/** Shader source storage — populated by setShaderSource() or bundler plugin */
const shaderSources: { primal: string; dual: string } = { primal: '', dual: '' };

/**
 * Set WGSL shader source code. Call this before creating a GPUSolver2D.
 * In a Vite/bundler environment, import with ?raw and pass here.
 * In a standalone environment, fetch the .wgsl files and pass the text.
 */
export function setShaderSource(primal: string, dual: string): void {
  shaderSources.primal = primal;
  shaderSources.dual = dual;
}

// ─── GPU Buffer Layout Constants ────────────────────────────────────────────

/** Body state: [x, y, angle, vx, vy, omega, mass, inertia] = 8 floats */
const BODY_STRIDE = 8;
/** Body prev: [prev_x, prev_y, prev_angle, inertial_x, inertial_y, inertial_angle, 0, 0] = 8 floats */
const BODY_PREV_STRIDE = 8;
/** Constraint row GPU struct size in bytes (must match WGSL struct) */
const CONSTRAINT_ROW_BYTES = 96; // 24 floats * 4 bytes (padded to 16-byte alignment)
/** Solver params uniform size */
const SOLVER_PARAMS_BYTES = 32; // 8 fields * 4 bytes
const WORKGROUP_SIZE = 64;

// ─── GPU Solver ─────────────────────────────────────────────────────────────

export class GPUSolver2D {
  config: SolverConfig;
  bodyStore: BodyStore2D;
  constraintStore: ConstraintStore;
  ignorePairs: Set<string> = new Set();
  jointConstraintIndices: number[] = [];
  colorGroups: ColorGroup[] = [];

  private gpu: GPUContext;
  private initialized = false;

  // GPU resources
  private bodyStateBuffer!: GPUBuffer;
  private bodyPrevBuffer!: GPUBuffer;
  private constraintBuffer!: GPUBuffer;
  private solverParamsBuffer!: GPUBuffer;
  private colorIndicesBuffer!: GPUBuffer;
  private bodyConstraintRangesBuffer!: GPUBuffer;
  private readbackBuffer!: GPUBuffer;

  private primalPipeline!: GPUComputePipeline;
  private dualPipeline!: GPUComputePipeline;
  private primalBindGroupLayout!: GPUBindGroupLayout;
  private dualBindGroupLayout!: GPUBindGroupLayout;

  // Tracks max allocated sizes to avoid reallocation
  private maxBodies = 0;
  private maxConstraints = 0;

  constructor(gpu: GPUContext, config: Partial<SolverConfig> = {}) {
    this.gpu = gpu;
    this.config = { ...DEFAULT_SOLVER_CONFIG_2D, ...config };
    this.bodyStore = new BodyStore2D();
    this.constraintStore = new ConstraintStore();
  }

  /** Initialize GPU pipelines. Must be called once before stepping. */
  init(): void {
    if (this.initialized) return;
    const device = this.gpu.device;

    // ─── Primal pipeline layout ───────────────────────────────────
    this.primalBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    if (!shaderSources.primal || !shaderSources.dual) {
      throw new Error(
        'WGSL shader source not set. Call setShaderSource(primalWGSL, dualWGSL) before init(). ' +
        'In Vite: import primal from "./shaders/primal-update-2d.wgsl?raw"'
      );
    }

    const primalModule = device.createShaderModule({ code: shaderSources.primal });
    this.primalPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.primalBindGroupLayout],
      }),
      compute: { module: primalModule, entryPoint: 'main' },
    });

    // ─── Dual pipeline layout ─────────────────────────────────────
    this.dualBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const dualModule = device.createShaderModule({ code: shaderSources.dual });
    this.dualPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.dualBindGroupLayout],
      }),
      compute: { module: dualModule, entryPoint: 'main' },
    });

    // ─── Solver params uniform (fixed size) ───────────────────────
    this.solverParamsBuffer = device.createBuffer({
      size: SOLVER_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.initialized = true;
  }

  /**
   * Perform one physics timestep using GPU compute shaders.
   *
   * CPU handles: broadphase, narrowphase, graph coloring, buffer upload.
   * GPU handles: primal updates (per color group), dual updates.
   * CPU handles: velocity recovery from readback data.
   */
  async step(): Promise<void> {
    if (!this.initialized) this.init();

    const { config, bodyStore, constraintStore, gpu } = this;
    const dt = config.dt;
    const gravity = config.gravity as Vec2;
    const bodies = bodyStore.bodies;
    const device = gpu.device;

    if (bodies.length === 0) return;

    // ─── 1. CPU: Collision Detection ──────────────────────────────
    constraintStore.clearContacts();

    for (let i = 0; i < bodies.length; i++) {
      const a = bodies[i];
      const aabbA = bodyStore.getAABB(a);
      for (let j = i + 1; j < bodies.length; j++) {
        const b = bodies[j];
        if (a.type !== RigidBodyType.Dynamic && b.type !== RigidBodyType.Dynamic) continue;
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (this.ignorePairs.has(key)) continue;
        if (!aabb2DOverlap(aabbA, bodyStore.getAABB(b))) continue;
        const manifold = collide2D(a, b);
        if (manifold) {
          const rows = createContactConstraintRows(manifold, a, b, config.penaltyMin, Infinity);
          constraintStore.addRows(rows);
        }
      }
    }

    constraintStore.warmstartContacts();

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

    // Initialize bodies (save prev, compute inertial targets)
    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;
      body.angularVelocity = Math.max(-50, Math.min(50, body.angularVelocity));
      body.prevPosition = { ...body.position };
      body.prevAngle = body.angle;

      let vx = body.velocity.x;
      let vy = body.velocity.y;
      let omega = body.angularVelocity;
      if (body.linearDamping > 0) {
        const f = 1 / (1 + body.linearDamping * dt);
        vx *= f; vy *= f;
      }
      if (body.angularDamping > 0) {
        omega *= 1 / (1 + body.angularDamping * dt);
      }

      body.inertialPosition = {
        x: body.position.x + vx * dt + gravity.x * body.gravityScale * dt * dt,
        y: body.position.y + vy * dt + gravity.y * body.gravityScale * dt * dt,
      };
      body.inertialAngle = body.angle + omega * dt;
      body.prevVelocity = { ...body.velocity };
    }

    // ─── 3. CPU→GPU: Upload Buffers ───────────────────────────────
    const numBodies = bodies.length;
    const numConstraints = constraintStore.rows.length;

    this.ensureBuffers(numBodies, numConstraints);

    // Upload body state: [x, y, angle, vx, vy, omega, mass, inertia]
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
    device.queue.writeBuffer(this.bodyStateBuffer, 0, bodyData);

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
    }
    device.queue.writeBuffer(this.bodyPrevBuffer, 0, prevData);

    // Upload constraints
    if (numConstraints > 0) {
      const crData = new Float32Array(numConstraints * 24); // 24 floats per constraint
      for (let i = 0; i < numConstraints; i++) {
        const row = constraintStore.rows[i];
        const off = i * 24;
        // Match the WGSL ConstraintRow struct layout
        const view = new DataView(crData.buffer, off * 4, 24 * 4);
        view.setInt32(0, row.bodyA, true);
        view.setInt32(4, row.bodyB, true);
        view.setUint32(8, row.type, true);
        view.setUint32(12, 0, true); // padding
        // jacobian_a (vec4)
        view.setFloat32(16, row.jacobianA[0], true);
        view.setFloat32(20, row.jacobianA[1], true);
        view.setFloat32(24, row.jacobianA[2], true);
        view.setFloat32(28, 0, true);
        // jacobian_b (vec4)
        view.setFloat32(32, row.jacobianB[0], true);
        view.setFloat32(36, row.jacobianB[1], true);
        view.setFloat32(40, row.jacobianB[2], true);
        view.setFloat32(44, 0, true);
        // hessian_diag_a (vec4)
        view.setFloat32(48, row.hessianDiagA[0], true);
        view.setFloat32(52, row.hessianDiagA[1], true);
        view.setFloat32(56, row.hessianDiagA[2], true);
        view.setFloat32(60, 0, true);
        // hessian_diag_b (vec4)
        view.setFloat32(64, row.hessianDiagB[0], true);
        view.setFloat32(68, row.hessianDiagB[1], true);
        view.setFloat32(72, row.hessianDiagB[2], true);
        view.setFloat32(76, 0, true);
        // scalars
        view.setFloat32(80, row.c, true);
        view.setFloat32(84, row.c0, true);
        view.setFloat32(88, row.lambda, true);
        view.setFloat32(92, row.penalty, true);
      }
      device.queue.writeBuffer(this.constraintBuffer, 0, crData);
    }

    // Build body→constraint index ranges
    const bodyConstraintRanges = new Uint32Array(numBodies * 2);
    // Simple approach: for each body, find all constraint rows involving it
    // This is O(N*K) but fine for CPU-side prep
    const constraintIndicesPerBody: number[][] = Array.from({ length: numBodies }, () => []);
    for (let ci = 0; ci < numConstraints; ci++) {
      const row = constraintStore.rows[ci];
      if (!row.active) continue;
      if (row.bodyA >= 0 && row.bodyA < numBodies) constraintIndicesPerBody[row.bodyA].push(ci);
      if (row.bodyB >= 0 && row.bodyB < numBodies) constraintIndicesPerBody[row.bodyB].push(ci);
    }
    // Flatten into a global index array + ranges
    const flatConstraintIndices: number[] = [];
    for (let i = 0; i < numBodies; i++) {
      bodyConstraintRanges[i * 2 + 0] = flatConstraintIndices.length;
      bodyConstraintRanges[i * 2 + 1] = constraintIndicesPerBody[i].length;
      flatConstraintIndices.push(...constraintIndicesPerBody[i]);
    }
    device.queue.writeBuffer(this.bodyConstraintRangesBuffer, 0, bodyConstraintRanges);

    // ─── 4. GPU: Solver Iterations ────────────────────────────────
    const totalIterations = config.postStabilize ? config.iterations + 1 : config.iterations;

    for (let iter = 0; iter < totalIterations; iter++) {
      const isStabilization = config.postStabilize && iter === totalIterations - 1;

      // Upload solver params for this iteration
      const paramsData = new Float32Array(8);
      paramsData[0] = dt;
      paramsData[1] = (gravity as Vec2).x;
      paramsData[2] = (gravity as Vec2).y;
      paramsData[3] = config.penaltyMin;
      paramsData[4] = config.penaltyMax;
      const paramsView = new DataView(paramsData.buffer);
      paramsView.setUint32(20, numBodies, true);
      paramsView.setUint32(24, numConstraints, true);
      paramsView.setUint32(28, isStabilization ? 1 : 0, true);
      device.queue.writeBuffer(this.solverParamsBuffer, 0, paramsData);

      const encoder = device.createCommandEncoder();

      // ─── 4a. Primal update: dispatch per color group ──────────
      for (const colorGroup of this.colorGroups) {
        if (colorGroup.bodyIndices.length === 0) continue;

        // Upload color group body indices
        const colorData = new Uint32Array(colorGroup.bodyIndices);
        device.queue.writeBuffer(this.colorIndicesBuffer, 0, colorData);

        // Create bind group for this dispatch
        const bindGroup = device.createBindGroup({
          layout: this.primalBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.solverParamsBuffer } },
            { binding: 1, resource: { buffer: this.bodyStateBuffer } },
            { binding: 2, resource: { buffer: this.bodyPrevBuffer } },
            { binding: 3, resource: { buffer: this.constraintBuffer } },
            { binding: 4, resource: { buffer: this.colorIndicesBuffer } },
            { binding: 5, resource: { buffer: this.bodyConstraintRangesBuffer } },
          ],
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.primalPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(colorGroup.bodyIndices.length / WORKGROUP_SIZE));
        pass.end();
      }

      // ─── 4b. Dual update: all constraints ─────────────────────
      if (!isStabilization && numConstraints > 0) {
        // Dual params need beta
        const dualParams = new Float32Array(8);
        dualParams[0] = dt;
        dualParams[1] = (gravity as Vec2).x;
        dualParams[2] = (gravity as Vec2).y;
        dualParams[3] = config.penaltyMin;
        dualParams[4] = config.penaltyMax;
        const dualView = new DataView(dualParams.buffer);
        dualView.setUint32(20, numBodies, true);
        dualView.setUint32(24, numConstraints, true);
        dualParams[7] = config.beta;
        device.queue.writeBuffer(this.solverParamsBuffer, 0, dualParams);

        const dualBindGroup = device.createBindGroup({
          layout: this.dualBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.solverParamsBuffer } },
            { binding: 1, resource: { buffer: this.bodyStateBuffer } },
            { binding: 2, resource: { buffer: this.bodyPrevBuffer } },
            { binding: 3, resource: { buffer: this.constraintBuffer } },
          ],
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.dualPipeline);
        pass.setBindGroup(0, dualBindGroup);
        pass.dispatchWorkgroups(Math.ceil(numConstraints / WORKGROUP_SIZE));
        pass.end();
      }

      device.queue.submit([encoder.finish()]);
    }

    // ─── 5. GPU→CPU: Read Back Body Positions ─────────────────────
    const readbackSize = numBodies * BODY_STRIDE * 4;
    const stagingBuffer = device.createBuffer({
      size: readbackSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(this.bodyStateBuffer, 0, stagingBuffer, 0, readbackSize);
    device.queue.submit([copyEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    // ─── 6. CPU: Apply Results & Velocity Recovery ────────────────
    for (let i = 0; i < numBodies; i++) {
      const body = bodies[i];
      if (body.type !== RigidBodyType.Dynamic) continue;

      const off = i * BODY_STRIDE;
      body.position.x = resultData[off + 0];
      body.position.y = resultData[off + 1];
      body.angle = resultData[off + 2];

      // BDF1 velocity recovery
      body.velocity = {
        x: (body.position.x - body.prevPosition.x) / dt,
        y: (body.position.y - body.prevPosition.y) / dt,
      };
      body.angularVelocity = (body.angle - body.prevAngle) / dt;
    }

    // Also read back constraint data for warmstarting next frame
    if (numConstraints > 0) {
      const crReadbackSize = numConstraints * 24 * 4;
      const crStaging = device.createBuffer({
        size: crReadbackSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const crEncoder = device.createCommandEncoder();
      crEncoder.copyBufferToBuffer(this.constraintBuffer, 0, crStaging, 0, crReadbackSize);
      device.queue.submit([crEncoder.finish()]);

      await crStaging.mapAsync(GPUMapMode.READ);
      const crResult = new DataView(crStaging.getMappedRange().slice(0));
      crStaging.unmap();
      crStaging.destroy();

      // Update CPU-side constraint lambdas and penalties from GPU results
      for (let i = 0; i < numConstraints; i++) {
        const off = i * 24 * 4;
        constraintStore.rows[i].lambda = crResult.getFloat32(off + 88, true);
        constraintStore.rows[i].penalty = crResult.getFloat32(off + 92, true);
      }
    }
  }

  /** Ensure GPU buffers are large enough for current body/constraint count */
  private ensureBuffers(numBodies: number, numConstraints: number): void {
    const device = this.gpu.device;

    if (numBodies > this.maxBodies) {
      this.maxBodies = Math.max(numBodies, this.maxBodies * 2, 64);
      if (this.bodyStateBuffer) this.bodyStateBuffer.destroy();
      if (this.bodyPrevBuffer) this.bodyPrevBuffer.destroy();
      if (this.bodyConstraintRangesBuffer) this.bodyConstraintRangesBuffer.destroy();
      if (this.colorIndicesBuffer) this.colorIndicesBuffer.destroy();

      const bodyBytes = this.maxBodies * BODY_STRIDE * 4;
      this.bodyStateBuffer = device.createBuffer({
        size: bodyBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.bodyPrevBuffer = device.createBuffer({
        size: bodyBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.bodyConstraintRangesBuffer = device.createBuffer({
        size: this.maxBodies * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.colorIndicesBuffer = device.createBuffer({
        size: this.maxBodies * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }

    const neededConstraints = Math.max(numConstraints, 1);
    if (neededConstraints > this.maxConstraints) {
      this.maxConstraints = Math.max(neededConstraints, this.maxConstraints * 2, 64);
      if (this.constraintBuffer) this.constraintBuffer.destroy();
      this.constraintBuffer = device.createBuffer({
        size: this.maxConstraints * 24 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
  }

  /** Destroy all GPU resources */
  destroy(): void {
    if (this.bodyStateBuffer) this.bodyStateBuffer.destroy();
    if (this.bodyPrevBuffer) this.bodyPrevBuffer.destroy();
    if (this.constraintBuffer) this.constraintBuffer.destroy();
    if (this.solverParamsBuffer) this.solverParamsBuffer.destroy();
    if (this.colorIndicesBuffer) this.colorIndicesBuffer.destroy();
    if (this.bodyConstraintRangesBuffer) this.bodyConstraintRangesBuffer.destroy();
  }
}
