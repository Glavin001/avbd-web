/**
 * GPU LBVH (Linear Bounding Volume Hierarchy) — Karras 2012 algorithm.
 *
 * Builds a BVH on the GPU for broad-phase collision detection.
 * Pipeline per frame:
 *   1. Compute AABBs + Morton codes from body state
 *   2. Radix sort Morton codes + indices
 *   3. Build BVH topology (Karras 2012)
 *   4. Bottom-up AABB refit (atomic counters)
 *   5. BVH traversal → collision pair buffer
 *
 * Supports both 2D (4-float AABBs) and 3D (6-float AABBs).
 */

import { GPUContext } from './gpu-context.js';
import { GpuRadixSort } from './gpu-radix-sort.js';
import {
  MORTON_CODES_2D_WGSL,
  MORTON_CODES_3D_WGSL,
  BVH_BUILD_WGSL,
  BVH_REFIT_2D_WGSL,
  BVH_REFIT_3D_WGSL,
  BVH_TRAVERSE_2D_WGSL,
  BVH_TRAVERSE_3D_WGSL,
} from '../shaders/embedded.js';

const WG_SIZE = 256;

/** Dimension mode for the BVH. */
export type BvhMode = '2d' | '3d';

/**
 * Manages GPU buffers and pipelines for LBVH construction + traversal.
 */
export class GpuBvh {
  private ctx: GPUContext;
  private mode: BvhMode;
  private radixSort: GpuRadixSort;

  // Pipeline objects
  private mortonPipeline!: GPUComputePipeline;
  private buildPipeline!: GPUComputePipeline;
  private refitPipeline!: GPUComputePipeline;
  private traversePipeline!: GPUComputePipeline;

  // Bind group layouts
  private mortonLayout!: GPUBindGroupLayout;
  private buildLayout!: GPUBindGroupLayout;
  private refitLayout!: GPUBindGroupLayout;
  private traverseLayout!: GPUBindGroupLayout;

  // GPU Buffers (grow as needed)
  private aabbBuf!: GPUBuffer;        // per-body AABBs
  private mortonBuf!: GPUBuffer;      // Morton codes
  private indexBuf!: GPUBuffer;       // object indices (identity, then sorted)
  private leftChildBuf!: GPUBuffer;   // internal node left children
  private rightChildBuf!: GPUBuffer;  // internal node right children
  private parentBuf!: GPUBuffer;      // parent of each node
  private nodeAabbBuf!: GPUBuffer;    // per-node AABBs (internal + leaf)
  private visitCountBuf!: GPUBuffer;  // atomic visit counters for refit
  private pairBuf!: GPUBuffer;        // output collision pairs
  private pairCountBuf!: GPUBuffer;   // atomic pair counter (single u32)
  private pairCountStagingBuf!: GPUBuffer; // staging for readback

  // Uniform buffers
  private mortonParamsBuf!: GPUBuffer;
  private buildParamsBuf!: GPUBuffer;
  private refitParamsBuf!: GPUBuffer;
  private traverseParamsBuf!: GPUBuffer;

  private capacity = 0;
  private maxPairs = 0;
  private aabbStride: number; // 4 for 2D, 6 for 3D
  private _lastSortedVals: GPUBuffer | null = null;
  private _lastSortedKeys: GPUBuffer | null = null;

  /** Pre-allocated typed arrays for params upload. */
  private mortonParamsData!: Float32Array;
  private mortonParamsU32!: Uint32Array;

  constructor(ctx: GPUContext, mode: BvhMode) {
    this.ctx = ctx;
    this.mode = mode;
    this.aabbStride = mode === '2d' ? 4 : 6;
    this.radixSort = new GpuRadixSort(ctx);
    this.initPipelines();
  }

  private initPipelines(): void {
    const device = this.ctx.device;
    const is3D = this.mode === '3d';

    // ─── Morton Code Pipeline ───
    // Shader bindings: 0=body_state, 1=collider_info, 2=params(uniform), 3=morton_codes, 4=aabb_buffer
    this.mortonLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_state
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // collider_info
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // morton_codes
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // aabb_buffer
      ],
    });

    const mortonShader = device.createShaderModule({
      code: is3D ? MORTON_CODES_3D_WGSL : MORTON_CODES_2D_WGSL,
    });
    this.mortonPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.mortonLayout] }),
      compute: { module: mortonShader, entryPoint: is3D ? 'morton_codes_3d' : 'morton_codes_2d' },
    });

    // ─── BVH Build Pipeline ───
    // Shader bindings: 0=morton_codes, 1=params(uniform), 2=left_child, 3=right_child, 4=parent
    this.buildLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // morton_codes
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // left_child
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // right_child
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // parent
      ],
    });

    const buildShader = device.createShaderModule({ code: BVH_BUILD_WGSL });
    this.buildPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.buildLayout] }),
      compute: { module: buildShader, entryPoint: 'bvh_build' },
    });

    // ─── Refit Pipeline ───
    // Shader bindings: 0=sorted_indices, 1=aabb, 2=left, 3=right, 4=parent, 5=params(uniform), 6=node_aabb, 7=visit_count
    this.refitLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // sorted_indices
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // aabb_buffer
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // left_child
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // right_child
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // parent
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // node_aabb
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // visit_count
      ],
    });

    const refitShader = device.createShaderModule({
      code: is3D ? BVH_REFIT_3D_WGSL : BVH_REFIT_2D_WGSL,
    });
    this.refitPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.refitLayout] }),
      compute: { module: refitShader, entryPoint: is3D ? 'bvh_refit_3d' : 'bvh_refit_2d' },
    });

    // ─── Traversal Pipeline ───
    // Shader bindings: 0=node_aabb, 1=left, 2=right, 3=aabb, 4=sorted_indices, 5=body_types, 6=params(uniform), 7=pair_buffer, 8=pair_count
    this.traverseLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // node_aabb
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // left_child
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // right_child
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // aabb_buffer
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // sorted_indices
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // body_types
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // pair_buffer
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // pair_count
      ],
    });

    const traverseShader = device.createShaderModule({
      code: is3D ? BVH_TRAVERSE_3D_WGSL : BVH_TRAVERSE_2D_WGSL,
    });
    this.traversePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.traverseLayout] }),
      compute: { module: traverseShader, entryPoint: is3D ? 'bvh_traverse_3d' : 'bvh_traverse_2d' },
    });

    // ─── Uniform buffers ───
    this.mortonParamsBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.buildParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.refitParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.traverseParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Pre-allocate params arrays
    this.mortonParamsData = new Float32Array(8);
    this.mortonParamsU32 = new Uint32Array(this.mortonParamsData.buffer);
  }

  /**
   * Ensure all buffers are large enough for n bodies.
   */
  private ensureCapacity(n: number): void {
    if (n <= this.capacity) return;
    const device = this.ctx.device;
    const cap = Math.max(n, 64);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    // Destroy old
    if (this.aabbBuf) {
      this.aabbBuf.destroy();
      this.mortonBuf.destroy();
      this.indexBuf.destroy();
      this.leftChildBuf.destroy();
      this.rightChildBuf.destroy();
      this.parentBuf.destroy();
      this.nodeAabbBuf.destroy();
      this.visitCountBuf.destroy();
      this.pairBuf.destroy();
      this.pairCountBuf.destroy();
      this.pairCountStagingBuf.destroy();
    }

    const stride = this.aabbStride;
    const numInternal = Math.max(cap - 1, 1);
    const totalNodes = cap + numInternal;

    this.aabbBuf = device.createBuffer({ size: cap * stride * 4, usage: storage });
    this.mortonBuf = device.createBuffer({ size: cap * 4, usage: storage });
    this.indexBuf = device.createBuffer({ size: cap * 4, usage: storage });
    this.leftChildBuf = device.createBuffer({ size: numInternal * 4, usage: storage });
    this.rightChildBuf = device.createBuffer({ size: numInternal * 4, usage: storage });
    this.parentBuf = device.createBuffer({ size: totalNodes * 4, usage: storage });
    this.nodeAabbBuf = device.createBuffer({ size: totalNodes * stride * 4, usage: storage });
    this.visitCountBuf = device.createBuffer({ size: numInternal * 4, usage: storage });

    // Pair buffer: dense scenes (pyramids, piles) can have many AABB overlaps per body.
    // Use a generous estimate to avoid silent pair loss.
    this.maxPairs = Math.max(cap * 32, 4096);
    this.pairBuf = device.createBuffer({ size: this.maxPairs * 2 * 4, usage: storage });
    this.pairCountBuf = device.createBuffer({ size: 4, usage: storage });
    this.pairCountStagingBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.capacity = cap;
  }

  /**
   * Run the complete broad-phase pipeline: AABB → Morton → Sort → Build → Refit → Traverse.
   *
   * @param encoder - Command encoder to record into
   * @param n - Number of bodies
   * @param bodyStateBuf - GPU buffer with body state data
   * @param colliderInfoBuf - GPU buffer with collider shape info
   * @param bodyTypesBuf - GPU buffer with body types (u32 per body)
   * @param sceneBounds - [minX, minY, (minZ,) maxX, maxY, (maxZ)] scene extents
   * @returns The pair buffer and pair count buffer (read pair_count for actual count)
   */
  buildAndTraverse(
    encoder: GPUCommandEncoder,
    n: number,
    bodyStateBuf: GPUBuffer,
    colliderInfoBuf: GPUBuffer,
    bodyTypesBuf: GPUBuffer,
    sceneBounds: Float32Array,
  ): { pairBuffer: GPUBuffer; pairCountBuffer: GPUBuffer; pairCountStaging: GPUBuffer } {
    if (n <= 1) {
      this.ensureCapacity(2); // ensure buffers exist
      const zero = new Uint32Array([0]);
      (this.ctx.device.queue as any).writeBuffer(this.pairCountBuf, 0, zero);
      return {
        pairBuffer: this.pairBuf,
        pairCountBuffer: this.pairCountBuf,
        pairCountStaging: this.pairCountStagingBuf,
      };
    }

    this.ensureCapacity(n);
    const device = this.ctx.device;
    const numWG = Math.ceil(n / WG_SIZE);
    const numInternal = n - 1;

    // ─── 1. Morton Code Generation + AABB computation ───
    const is3D = this.mode === '3d';
    this.mortonParamsU32[0] = n;
    if (is3D) {
      this.mortonParamsData[1] = sceneBounds[0]; // minX
      this.mortonParamsData[2] = sceneBounds[1]; // minY
      this.mortonParamsData[3] = sceneBounds[2]; // minZ
      this.mortonParamsData[4] = sceneBounds[3]; // maxX
      this.mortonParamsData[5] = sceneBounds[4]; // maxY
      this.mortonParamsData[6] = sceneBounds[5]; // maxZ
    } else {
      this.mortonParamsData[1] = sceneBounds[0]; // minX
      this.mortonParamsData[2] = sceneBounds[1]; // minY
      this.mortonParamsData[3] = sceneBounds[2]; // maxX
      this.mortonParamsData[4] = sceneBounds[3]; // maxY
    }
    (device.queue as any).writeBuffer(this.mortonParamsBuf, 0, this.mortonParamsData);

    // Initialize identity index buffer on CPU (morton shader doesn't output indices)
    const identityIndices = new Uint32Array(n);
    for (let i = 0; i < n; i++) identityIndices[i] = i;
    (device.queue as any).writeBuffer(this.indexBuf, 0, identityIndices);

    // Morton shader: bindings 0=body_state, 1=collider_info, 2=params, 3=morton, 4=aabb
    const mortonBindGroup = device.createBindGroup({
      layout: this.mortonLayout,
      entries: [
        { binding: 0, resource: { buffer: bodyStateBuf } },
        { binding: 1, resource: { buffer: colliderInfoBuf } },
        { binding: 2, resource: { buffer: this.mortonParamsBuf } },
        { binding: 3, resource: { buffer: this.mortonBuf } },
        { binding: 4, resource: { buffer: this.aabbBuf } },
      ],
    });

    let pass = encoder.beginComputePass();
    pass.setPipeline(this.mortonPipeline);
    pass.setBindGroup(0, mortonBindGroup);
    pass.dispatchWorkgroups(numWG);
    pass.end();

    // ─── 2. Radix Sort ───
    const { sortedKeys, sortedVals } = this.radixSort.sort(encoder, n, this.mortonBuf, this.indexBuf);
    this._lastSortedVals = sortedVals;
    this._lastSortedKeys = sortedKeys;

    // ─── 3. BVH Build (Karras topology) ───
    if (numInternal > 0) {
      const buildParams = new Uint32Array([n, 0, 0, 0]);
      (device.queue as any).writeBuffer(this.buildParamsBuf, 0, buildParams);

      // Initialize parent array to -1
      const parentInit = new Int32Array(n + numInternal).fill(-1);
      (device.queue as any).writeBuffer(this.parentBuf, 0, parentInit);

      // Build shader: bindings 0=morton, 1=params, 2=left, 3=right, 4=parent
      const buildBindGroup = device.createBindGroup({
        layout: this.buildLayout,
        entries: [
          { binding: 0, resource: { buffer: sortedKeys } },
          { binding: 1, resource: { buffer: this.buildParamsBuf } },
          { binding: 2, resource: { buffer: this.leftChildBuf } },
          { binding: 3, resource: { buffer: this.rightChildBuf } },
          { binding: 4, resource: { buffer: this.parentBuf } },
        ],
      });

      pass = encoder.beginComputePass();
      pass.setPipeline(this.buildPipeline);
      pass.setBindGroup(0, buildBindGroup);
      pass.dispatchWorkgroups(Math.ceil(numInternal / WG_SIZE));
      pass.end();
    }

    // ─── 4. Bottom-Up Refit ───
    {
      // Clear visit counters
      const zeroVisit = new Uint32Array(Math.max(numInternal, 1)).fill(0);
      (device.queue as any).writeBuffer(this.visitCountBuf, 0, zeroVisit);

      const refitParams = new Uint32Array([n, numInternal, 0, 0]);
      (device.queue as any).writeBuffer(this.refitParamsBuf, 0, refitParams);

      // Refit shader: bindings 0=sorted_indices, 1=aabb, 2=left, 3=right, 4=parent, 5=params, 6=node_aabb, 7=visit
      const refitBindGroup = device.createBindGroup({
        layout: this.refitLayout,
        entries: [
          { binding: 0, resource: { buffer: sortedVals } },
          { binding: 1, resource: { buffer: this.aabbBuf } },
          { binding: 2, resource: { buffer: this.leftChildBuf } },
          { binding: 3, resource: { buffer: this.rightChildBuf } },
          { binding: 4, resource: { buffer: this.parentBuf } },
          { binding: 5, resource: { buffer: this.refitParamsBuf } },
          { binding: 6, resource: { buffer: this.nodeAabbBuf } },
          { binding: 7, resource: { buffer: this.visitCountBuf } },
        ],
      });

      pass = encoder.beginComputePass();
      pass.setPipeline(this.refitPipeline);
      pass.setBindGroup(0, refitBindGroup);
      pass.dispatchWorkgroups(numWG); // one thread per leaf
      pass.end();
    }

    // ─── 5. BVH Traversal ───
    {
      // Clear pair count
      const zeroPairCount = new Uint32Array([0]);
      (device.queue as any).writeBuffer(this.pairCountBuf, 0, zeroPairCount);

      const traverseParams = new Uint32Array([n, numInternal, this.maxPairs, 0]);
      (device.queue as any).writeBuffer(this.traverseParamsBuf, 0, traverseParams);

      // Traverse shader: bindings 0=node_aabb, 1=left, 2=right, 3=aabb, 4=sorted, 5=body_types, 6=params, 7=pair, 8=pair_count
      const traverseBindGroup = device.createBindGroup({
        layout: this.traverseLayout,
        entries: [
          { binding: 0, resource: { buffer: this.nodeAabbBuf } },
          { binding: 1, resource: { buffer: this.leftChildBuf } },
          { binding: 2, resource: { buffer: this.rightChildBuf } },
          { binding: 3, resource: { buffer: this.aabbBuf } },
          { binding: 4, resource: { buffer: sortedVals } },
          { binding: 5, resource: { buffer: bodyTypesBuf } },
          { binding: 6, resource: { buffer: this.traverseParamsBuf } },
          { binding: 7, resource: { buffer: this.pairBuf } },
          { binding: 8, resource: { buffer: this.pairCountBuf } },
        ],
      });

      pass = encoder.beginComputePass();
      pass.setPipeline(this.traversePipeline);
      pass.setBindGroup(0, traverseBindGroup);
      pass.dispatchWorkgroups(numWG);
      pass.end();
    }

    // Copy pair count to staging for CPU readback
    encoder.copyBufferToBuffer(this.pairCountBuf, 0, this.pairCountStagingBuf, 0, 4);

    return {
      pairBuffer: this.pairBuf,
      pairCountBuffer: this.pairCountBuf,
      pairCountStaging: this.pairCountStagingBuf,
    };
  }

  /**
   * Read back the number of pairs found after GPU submission.
   */
  async readPairCount(): Promise<number> {
    try {
      await this.pairCountStagingBuf.mapAsync(GPUMapMode.READ);
    } catch (e) {
      console.error('GPU readPairCount failed:', (e as Error).message);
      return 0;
    }
    const count = new Uint32Array(this.pairCountStagingBuf.getMappedRange().slice(0))[0];
    this.pairCountStagingBuf.unmap();
    return Math.min(count, this.maxPairs);
  }

  /** Read back actual pair buffer contents for diagnostics. */
  async readPairBuffer(numPairs: number): Promise<Array<[number, number]>> {
    const device = this.ctx.device;
    const byteSize = numPairs * 2 * 4;
    const staging = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.pairBuf, 0, staging, 0, byteSize);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const pairs: Array<[number, number]> = [];
    for (let i = 0; i < numPairs; i++) {
      pairs.push([data[i * 2], data[i * 2 + 1]]);
    }
    return pairs;
  }

  /** Get the pair buffer for use in narrowphase. */
  getPairBuffer(): GPUBuffer { return this.pairBuf; }

  /** Get the AABB buffer. */
  getAabbBuffer(): GPUBuffer { return this.aabbBuf; }

  /** Read back all BVH diagnostic data for debugging. */
  async readDiagnostics(n: number): Promise<{
    aabbs: Float32Array;
    nodeAabbs: Float32Array;
    sortedIndices: Uint32Array;
    sortedKeys: Uint32Array;
    mortonCodes: Uint32Array;
    leftChildren: Int32Array;
    rightChildren: Int32Array;
    parents: Int32Array;
    visitCounts: Uint32Array;
  }> {
    const device = this.ctx.device;
    const stride = this.aabbStride;
    const numInternal = n - 1;
    const totalNodes = n + numInternal;

    const readBuffer = async (src: GPUBuffer, size: number): Promise<ArrayBuffer> => {
      const staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const data = staging.getMappedRange().slice(0);
      staging.unmap();
      staging.destroy();
      return data;
    };

    const [aabbData, nodeAabbData, sortedData, sortedKeysData, mortonData, leftData, rightData, parentData, visitData] =
      await Promise.all([
        readBuffer(this.aabbBuf, n * stride * 4),
        readBuffer(this.nodeAabbBuf, totalNodes * stride * 4),
        readBuffer(this._lastSortedVals ?? this.indexBuf, n * 4),
        readBuffer(this._lastSortedKeys ?? this.mortonBuf, n * 4),
        readBuffer(this.mortonBuf, n * 4),
        readBuffer(this.leftChildBuf, numInternal * 4),
        readBuffer(this.rightChildBuf, numInternal * 4),
        readBuffer(this.parentBuf, totalNodes * 4),
        readBuffer(this.visitCountBuf, numInternal * 4),
      ]);

    return {
      aabbs: new Float32Array(aabbData),
      nodeAabbs: new Float32Array(nodeAabbData),
      sortedIndices: new Uint32Array(sortedData),
      sortedKeys: new Uint32Array(sortedKeysData),
      mortonCodes: new Uint32Array(mortonData),
      leftChildren: new Int32Array(leftData),
      rightChildren: new Int32Array(rightData),
      parents: new Int32Array(parentData),
      visitCounts: new Uint32Array(visitData),
    };
  }

  destroy(): void {
    this.radixSort.destroy();
    this.aabbBuf?.destroy();
    this.mortonBuf?.destroy();
    this.indexBuf?.destroy();
    this.leftChildBuf?.destroy();
    this.rightChildBuf?.destroy();
    this.parentBuf?.destroy();
    this.nodeAabbBuf?.destroy();
    this.visitCountBuf?.destroy();
    this.pairBuf?.destroy();
    this.pairCountBuf?.destroy();
    this.pairCountStagingBuf?.destroy();
    this.mortonParamsBuf?.destroy();
    this.buildParamsBuf?.destroy();
    this.refitParamsBuf?.destroy();
    this.traverseParamsBuf?.destroy();
  }
}
