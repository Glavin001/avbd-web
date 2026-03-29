/**
 * GPU Narrow Phase Collision Detection + Constraint Assembly.
 *
 * Takes collision pairs from BVH traversal, runs shape-specific narrow phase
 * on GPU, then assembles constraint rows for the AVBD solver.
 *
 * Pipeline:
 *   1. Narrow phase: per-pair contact generation (SAT, distance checks)
 *   2. Constraint assembly: convert contacts to solver constraint rows
 *   3. Contact persistence: warm-start from cached contacts via feature IDs
 */

import { GPUContext } from './gpu-context.js';
import type { BvhMode } from './gpu-bvh.js';
import {
  NARROWPHASE_2D_WGSL,
  NARROWPHASE_3D_WGSL,
  CONSTRAINT_ASSEMBLY_2D_WGSL,
  CONSTRAINT_ASSEMBLY_3D_WGSL,
} from '../shaders/embedded.js';

const WG_SIZE = 256;

/** 2D contact: 10 u32s (40 bytes). 3D contact: 12 u32s (48 bytes). */
const CONTACT_STRIDE_2D = 10;
const CONTACT_STRIDE_3D = 12;

/** 2D constraint row: 28 floats. 3D: 36 floats. */
const CONSTRAINT_STRIDE_2D = 28;
const CONSTRAINT_STRIDE_3D = 36;

/** Max contacts per pair: 2 for 2D, 4 for 3D. */
const MAX_CONTACTS_PER_PAIR_2D = 2;
const MAX_CONTACTS_PER_PAIR_3D = 4;

/** Rows per contact: 2 for 2D (normal+friction), 3 for 3D (normal+2 friction). */
const ROWS_PER_CONTACT_2D = 2;
const ROWS_PER_CONTACT_3D = 3;

export class GpuNarrowphase {
  private ctx: GPUContext;
  private mode: BvhMode;

  // Pipelines
  private narrowphasePipeline!: GPUComputePipeline;
  private constraintAssemblyPipeline!: GPUComputePipeline;

  // Bind group layouts
  private narrowphaseLayout!: GPUBindGroupLayout;
  private assemblyLayout!: GPUBindGroupLayout;

  // Buffers
  private contactBuf!: GPUBuffer;
  private contactCountBuf!: GPUBuffer;
  private contactCountStagingBuf!: GPUBuffer;
  private constraintBuf!: GPUBuffer;
  private constraintCountBuf!: GPUBuffer;
  private constraintCountStagingBuf!: GPUBuffer;

  // Uniform buffers
  private narrowphaseParamsBuf!: GPUBuffer;
  private assemblyParamsBuf!: GPUBuffer;

  // Warm-start cache buffers
  private warmstartKeysBuf!: GPUBuffer;   // (bodyA, bodyB, featureId) per cached contact
  private warmstartValsBuf!: GPUBuffer;   // (penalty, lambda_n, lambda_t1, lambda_t2) per cached contact
  private warmstartAgeBuf!: GPUBuffer;    // age counter per cached contact

  private maxContacts = 0;
  private maxConstraints = 0;
  private maxPairs = 0;
  private warmstartCacheSize = 0;
  private _contactClearBuf: Uint32Array | null = null;

  private contactStride: number;
  private constraintStride: number;
  private maxContactsPerPair: number;
  private rowsPerContact: number;

  constructor(ctx: GPUContext, mode: BvhMode) {
    this.ctx = ctx;
    this.mode = mode;
    this.contactStride = mode === '2d' ? CONTACT_STRIDE_2D : CONTACT_STRIDE_3D;
    this.constraintStride = mode === '2d' ? CONSTRAINT_STRIDE_2D : CONSTRAINT_STRIDE_3D;
    this.maxContactsPerPair = mode === '2d' ? MAX_CONTACTS_PER_PAIR_2D : MAX_CONTACTS_PER_PAIR_3D;
    this.rowsPerContact = mode === '2d' ? ROWS_PER_CONTACT_2D : ROWS_PER_CONTACT_3D;
    this.initPipelines();
  }

  private initPipelines(): void {
    const device = this.ctx.device;
    const is3D = this.mode === '3d';

    // ─── Narrow Phase Pipeline ───
    this.narrowphaseLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // pair_buffer
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_state
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // collider_info
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // contact_buffer
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // contact_count
      ],
    });

    const npShader = device.createShaderModule({
      code: is3D ? NARROWPHASE_3D_WGSL : NARROWPHASE_2D_WGSL,
    });
    this.narrowphasePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.narrowphaseLayout] }),
      compute: { module: npShader, entryPoint: is3D ? 'narrowphase_3d' : 'narrowphase_2d' },
    });

    // ─── Constraint Assembly Pipeline ───
    this.assemblyLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // contact_buffer
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_state
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // collider_info
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // constraint_buffer
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // constraint_count
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // warmstart_keys
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // warmstart_vals
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // warmstart_age
      ],
    });

    const asmShader = device.createShaderModule({
      code: is3D ? CONSTRAINT_ASSEMBLY_3D_WGSL : CONSTRAINT_ASSEMBLY_2D_WGSL,
    });
    this.constraintAssemblyPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.assemblyLayout] }),
      compute: { module: asmShader, entryPoint: is3D ? 'constraint_assembly_3d' : 'constraint_assembly_2d' },
    });

    // Uniform buffers
    this.narrowphaseParamsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.assemblyParamsBuf = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Ensure buffers are large enough.
   */
  private ensureCapacity(maxPairs: number): void {
    if (maxPairs <= this.maxPairs) return;
    const device = this.ctx.device;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    // Destroy old
    if (this.contactBuf) {
      this.contactBuf.destroy();
      this.contactCountBuf.destroy();
      this.contactCountStagingBuf.destroy();
      this.constraintBuf.destroy();
      this.constraintCountBuf.destroy();
      this.constraintCountStagingBuf.destroy();
      this.warmstartKeysBuf.destroy();
      this.warmstartValsBuf.destroy();
      this.warmstartAgeBuf.destroy();
    }

    this.maxPairs = maxPairs;
    this.maxContacts = maxPairs * this.maxContactsPerPair;
    this.maxConstraints = this.maxContacts * this.rowsPerContact;

    // Contact buffer
    this.contactBuf = device.createBuffer({
      size: Math.max(this.maxContacts * this.contactStride * 4, 4),
      usage: storage,
    });
    this.contactCountBuf = device.createBuffer({ size: 4, usage: storage });
    this.contactCountStagingBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Constraint buffer
    this.constraintBuf = device.createBuffer({
      size: Math.max(this.maxConstraints * this.constraintStride * 4, 4),
      usage: storage,
    });
    this.constraintCountBuf = device.createBuffer({ size: 4, usage: storage });
    this.constraintCountStagingBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Warm-start cache (fixed size, larger than max contacts for hash table headroom)
    this.warmstartCacheSize = Math.max(this.maxContacts * 2, 1024);
    this.warmstartKeysBuf = device.createBuffer({
      size: this.warmstartCacheSize * 3 * 4, // 3 u32 per entry
      usage: storage,
    });
    this.warmstartValsBuf = device.createBuffer({
      size: this.warmstartCacheSize * 4 * 4, // 4 f32 per entry
      usage: storage,
    });
    this.warmstartAgeBuf = device.createBuffer({
      size: this.warmstartCacheSize * 4, // 1 u32 per entry
      usage: storage,
    });

    this.maxPairs = maxPairs;
  }

  /**
   * Run narrow phase + constraint assembly on GPU.
   *
   * @param encoder - Command encoder
   * @param numPairs - Number of collision pairs (from BVH traversal)
   * @param pairBuf - Pair buffer from BVH
   * @param bodyStateBuf - Body state buffer
   * @param colliderInfoBuf - Collider info buffer
   * @param collisionMargin - Collision margin (typically 0.0005)
   * @param dt - Timestep
   * @param penaltyMin - Minimum penalty value
   * @param alpha - Stabilization parameter
   * @returns Constraint buffer and count for the solver
   */
  generateConstraints(
    encoder: GPUCommandEncoder,
    numPairs: number,
    pairBuf: GPUBuffer,
    bodyStateBuf: GPUBuffer,
    colliderInfoBuf: GPUBuffer,
    collisionMargin: number,
    dt: number,
    penaltyMin: number,
    alpha: number,
  ): { constraintBuffer: GPUBuffer; constraintCountBuffer: GPUBuffer; constraintCountStaging: GPUBuffer; contactBuffer: GPUBuffer; contactCountBuffer: GPUBuffer } {
    if (numPairs <= 0) {
      const zero = new Uint32Array([0]);
      (this.ctx.device.queue as any).writeBuffer(this.contactCountBuf, 0, zero);
      (this.ctx.device.queue as any).writeBuffer(this.constraintCountBuf, 0, zero);
      return {
        constraintBuffer: this.constraintBuf,
        constraintCountBuffer: this.constraintCountBuf,
        constraintCountStaging: this.constraintCountStagingBuf,
        contactBuffer: this.contactBuf,
        contactCountBuffer: this.contactCountBuf,
      };
    }

    this.ensureCapacity(numPairs);
    const device = this.ctx.device;

    // ─── 1. Narrow Phase ───
    {
      // Clear contact count
      (device.queue as any).writeBuffer(this.contactCountBuf, 0, new Uint32Array([0]));

      // Clear contact buffer with sentinel values (0xFFFFFFFF) to prevent
      // stale contacts from previous frames being processed by the assembly shader.
      // The assembly shader checks bodyA == 0xFFFFFFFF to skip invalid slots.
      if (!this._contactClearBuf || this._contactClearBuf.byteLength < this.maxContacts * this.contactStride * 4) {
        this._contactClearBuf = new Uint32Array(this.maxContacts * this.contactStride);
        this._contactClearBuf.fill(0xFFFFFFFF);
      }
      (device.queue as any).writeBuffer(this.contactBuf, 0, this._contactClearBuf);

      // Upload params
      const paramsData = new Float32Array(4);
      const paramsU32 = new Uint32Array(paramsData.buffer);
      paramsU32[0] = numPairs;
      paramsU32[1] = this.maxContacts;
      paramsData[2] = collisionMargin;
      (device.queue as any).writeBuffer(this.narrowphaseParamsBuf, 0, paramsData);

      const npBindGroup = device.createBindGroup({
        layout: this.narrowphaseLayout,
        entries: [
          { binding: 0, resource: { buffer: this.narrowphaseParamsBuf } },
          { binding: 1, resource: { buffer: pairBuf } },
          { binding: 2, resource: { buffer: bodyStateBuf } },
          { binding: 3, resource: { buffer: colliderInfoBuf } },
          { binding: 4, resource: { buffer: this.contactBuf } },
          { binding: 5, resource: { buffer: this.contactCountBuf } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.narrowphasePipeline);
      pass.setBindGroup(0, npBindGroup);
      pass.dispatchWorkgroups(Math.ceil(numPairs / WG_SIZE));
      pass.end();
    }

    // ─── 2. Constraint Assembly ───
    {
      // Clear constraint count
      (device.queue as any).writeBuffer(this.constraintCountBuf, 0, new Uint32Array([0]));

      // Upload assembly params
      const asmParams = new Float32Array(8);
      const asmU32 = new Uint32Array(asmParams.buffer);
      asmU32[0] = this.maxContacts; // max_contacts (actual count read from contactCountBuf atomically in shader)
      asmU32[1] = this.maxConstraints;
      asmParams[2] = collisionMargin;
      asmParams[3] = dt;
      asmParams[4] = penaltyMin;
      asmParams[5] = alpha;
      asmU32[6] = this.warmstartCacheSize;
      (device.queue as any).writeBuffer(this.assemblyParamsBuf, 0, asmParams);

      const asmBindGroup = device.createBindGroup({
        layout: this.assemblyLayout,
        entries: [
          { binding: 0, resource: { buffer: this.assemblyParamsBuf } },
          { binding: 1, resource: { buffer: this.contactBuf } },
          { binding: 2, resource: { buffer: bodyStateBuf } },
          { binding: 3, resource: { buffer: colliderInfoBuf } },
          { binding: 4, resource: { buffer: this.constraintBuf } },
          { binding: 5, resource: { buffer: this.constraintCountBuf } },
          { binding: 6, resource: { buffer: this.warmstartKeysBuf } },
          { binding: 7, resource: { buffer: this.warmstartValsBuf } },
          { binding: 8, resource: { buffer: this.warmstartAgeBuf } },
        ],
      });

      // Dispatch for max_contacts threads (shader checks contact_count internally)
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.constraintAssemblyPipeline);
      pass.setBindGroup(0, asmBindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.maxContacts / WG_SIZE));
      pass.end();
    }

    // Copy constraint count to staging
    encoder.copyBufferToBuffer(this.constraintCountBuf, 0, this.constraintCountStagingBuf, 0, 4);

    return {
      constraintBuffer: this.constraintBuf,
      constraintCountBuffer: this.constraintCountBuf,
      constraintCountStaging: this.constraintCountStagingBuf,
      contactBuffer: this.contactBuf,
      contactCountBuffer: this.contactCountBuf,
    };
  }

  /**
   * Read back constraint count after GPU submission.
   */
  async readConstraintCount(): Promise<number> {
    try {
      await this.constraintCountStagingBuf.mapAsync(GPUMapMode.READ);
    } catch (e) {
      console.error('GPU readConstraintCount failed:', (e as Error).message);
      return 0;
    }
    const count = new Uint32Array(this.constraintCountStagingBuf.getMappedRange().slice(0))[0];
    this.constraintCountStagingBuf.unmap();
    return Math.min(count, this.maxConstraints);
  }

  /** Get the constraint buffer for direct use by the solver. */
  getConstraintBuffer(): GPUBuffer { return this.constraintBuf; }
  getContactBuffer(): GPUBuffer { return this.contactBuf; }

  destroy(): void {
    this.contactBuf?.destroy();
    this.contactCountBuf?.destroy();
    this.contactCountStagingBuf?.destroy();
    this.constraintBuf?.destroy();
    this.constraintCountBuf?.destroy();
    this.constraintCountStagingBuf?.destroy();
    this.warmstartKeysBuf?.destroy();
    this.warmstartValsBuf?.destroy();
    this.warmstartAgeBuf?.destroy();
    this.narrowphaseParamsBuf?.destroy();
    this.assemblyParamsBuf?.destroy();
  }
}
