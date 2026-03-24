/**
 * GPU Radix Sort — 4-bit LSD radix sort for u32 key-value pairs.
 *
 * Uses three compute passes per digit (8 digits for 32-bit keys):
 *   1. Histogram: count digit frequencies per workgroup tile
 *   2. Prefix Sum: exclusive scan over global histogram
 *   3. Scatter: write keys+values to sorted positions
 *
 * Ping-pongs between two buffer pairs across digit passes.
 */

import { GPUContext } from './gpu-context.js';
import { RADIX_SORT_WGSL } from '../shaders/embedded.js';

const WG_SIZE = 256;
const RADIX = 16; // 4-bit digits
const NUM_DIGITS = 8; // 32 / 4

export class GpuRadixSort {
  private ctx: GPUContext;
  private histogramPipeline!: GPUComputePipeline;
  private prefixSumPipeline!: GPUComputePipeline;
  private scatterPipeline!: GPUComputePipeline;
  private bindGroupLayout!: GPUBindGroupLayout;

  // Persistent buffers (grow as needed)
  private keysA!: GPUBuffer;
  private keysB!: GPUBuffer;
  private valsA!: GPUBuffer;
  private valsB!: GPUBuffer;
  private histogramBuf!: GPUBuffer;
  private paramsBuf!: GPUBuffer;
  private capacity = 0;
  private maxWorkgroups = 0;

  constructor(ctx: GPUContext) {
    this.ctx = ctx;
    this.initPipelines();
  }

  private initPipelines(): void {
    const device = this.ctx.device;

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const shaderModule = device.createShaderModule({ code: RADIX_SORT_WGSL });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });

    this.histogramPipeline = device.createComputePipeline({
      layout,
      compute: { module: shaderModule, entryPoint: 'radix_histogram' },
    });
    this.prefixSumPipeline = device.createComputePipeline({
      layout,
      compute: { module: shaderModule, entryPoint: 'radix_prefix_sum' },
    });
    this.scatterPipeline = device.createComputePipeline({
      layout,
      compute: { module: shaderModule, entryPoint: 'radix_scatter' },
    });

    this.paramsBuf = device.createBuffer({
      size: 16, // 4 u32s
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Ensure buffers are large enough for n elements.
   */
  private ensureCapacity(n: number): void {
    if (n <= this.capacity) return;

    const device = this.ctx.device;
    const bufSize = Math.max(n, 64) * 4; // u32 = 4 bytes
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    // Destroy old buffers
    if (this.keysA) {
      this.keysA.destroy();
      this.keysB.destroy();
      this.valsA.destroy();
      this.valsB.destroy();
      this.histogramBuf.destroy();
    }

    this.keysA = device.createBuffer({ size: bufSize, usage });
    this.keysB = device.createBuffer({ size: bufSize, usage });
    this.valsA = device.createBuffer({ size: bufSize, usage });
    this.valsB = device.createBuffer({ size: bufSize, usage });

    const numWG = Math.ceil(n / WG_SIZE);
    this.maxWorkgroups = numWG;
    // Histogram: 16 digits * numWorkgroups
    const histSize = RADIX * numWG * 4;
    this.histogramBuf = device.createBuffer({
      size: Math.max(histSize, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.capacity = n;
  }

  /**
   * Sort key-value pairs on the GPU.
   * Keys and values are uploaded, sorted in-place, and the sorted buffers are returned.
   *
   * @param keys - u32 Morton codes or sort keys
   * @param values - u32 object indices
   * @returns The GPU buffers containing sorted keys and values (valid until next sort call)
   */
  sort(
    encoder: GPUCommandEncoder,
    n: number,
    keysBuffer: GPUBuffer,
    valsBuffer: GPUBuffer,
  ): { sortedKeys: GPUBuffer; sortedVals: GPUBuffer } {
    if (n <= 1) {
      return { sortedKeys: keysBuffer, sortedVals: valsBuffer };
    }

    this.ensureCapacity(n);
    const device = this.ctx.device;
    const numWG = Math.ceil(n / WG_SIZE);

    // Copy input to keysA/valsA
    encoder.copyBufferToBuffer(keysBuffer, 0, this.keysA, 0, n * 4);
    encoder.copyBufferToBuffer(valsBuffer, 0, this.valsA, 0, n * 4);

    // Params data (reusable typed array)
    const paramsData = new Uint32Array(4);
    paramsData[0] = n;
    paramsData[2] = numWG;

    let readKeys = this.keysA;
    let readVals = this.valsA;
    let writeKeys = this.keysB;
    let writeVals = this.valsB;

    for (let digit = 0; digit < NUM_DIGITS; digit++) {
      paramsData[1] = digit * 4; // digit_shift
      (device.queue as any).writeBuffer(this.paramsBuf, 0, paramsData);

      const bindGroup = device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: { buffer: readKeys } },
          { binding: 2, resource: { buffer: readVals } },
          { binding: 3, resource: { buffer: writeKeys } },
          { binding: 4, resource: { buffer: writeVals } },
          { binding: 5, resource: { buffer: this.histogramBuf } },
        ],
      });

      // Pass 1: Histogram
      const histPass = encoder.beginComputePass();
      histPass.setPipeline(this.histogramPipeline);
      histPass.setBindGroup(0, bindGroup);
      histPass.dispatchWorkgroups(numWG);
      histPass.end();

      // Pass 2: Prefix Sum (single workgroup)
      const prefixPass = encoder.beginComputePass();
      prefixPass.setPipeline(this.prefixSumPipeline);
      prefixPass.setBindGroup(0, bindGroup);
      prefixPass.dispatchWorkgroups(1);
      prefixPass.end();

      // Pass 3: Scatter
      const scatterPass = encoder.beginComputePass();
      scatterPass.setPipeline(this.scatterPipeline);
      scatterPass.setBindGroup(0, bindGroup);
      scatterPass.dispatchWorkgroups(numWG);
      scatterPass.end();

      // Ping-pong
      const tmpK = readKeys; readKeys = writeKeys; writeKeys = tmpK;
      const tmpV = readVals; readVals = writeVals; writeVals = tmpV;
    }

    // After 8 passes (even number), result is back in keysA/valsA
    return { sortedKeys: readKeys, sortedVals: readVals };
  }

  /**
   * Sort and read back results to CPU (for testing).
   */
  async sortAndReadback(
    keys: Uint32Array,
    values: Uint32Array,
  ): Promise<{ sortedKeys: Uint32Array; sortedVals: Uint32Array }> {
    const n = keys.length;
    this.ensureCapacity(n);
    const device = this.ctx.device;

    // Upload
    (device.queue as any).writeBuffer(this.keysA, 0, keys);
    (device.queue as any).writeBuffer(this.valsA, 0, values);

    const encoder = device.createCommandEncoder();
    const { sortedKeys, sortedVals } = this.sort(encoder, n, this.keysA, this.valsA);

    // Readback via staging buffers
    const stagingKeys = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const stagingVals = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(sortedKeys, 0, stagingKeys, 0, n * 4);
    encoder.copyBufferToBuffer(sortedVals, 0, stagingVals, 0, n * 4);

    device.queue.submit([encoder.finish()]);

    await stagingKeys.mapAsync(GPUMapMode.READ);
    await stagingVals.mapAsync(GPUMapMode.READ);

    const resultKeys = new Uint32Array(stagingKeys.getMappedRange().slice(0));
    const resultVals = new Uint32Array(stagingVals.getMappedRange().slice(0));

    stagingKeys.unmap();
    stagingVals.unmap();
    stagingKeys.destroy();
    stagingVals.destroy();

    return { sortedKeys: resultKeys, sortedVals: resultVals };
  }

  destroy(): void {
    this.keysA?.destroy();
    this.keysB?.destroy();
    this.valsA?.destroy();
    this.valsB?.destroy();
    this.histogramBuf?.destroy();
    this.paramsBuf?.destroy();
  }
}
