/**
 * WebGPU context manager for the AVBD physics engine.
 * Handles device initialization, buffer management, and compute pipeline creation.
 */

export interface GPUContextOptions {
  /** Power preference for adapter selection */
  powerPreference?: GPUPowerPreference;
  /** Required WebGPU features */
  requiredFeatures?: GPUFeatureName[];
}

export class GPUContext {
  device!: GPUDevice;
  adapter!: GPUAdapter;
  /** Callback invoked when the GPU device is lost. Set by consumers to handle device loss. */
  onDeviceLost: ((message: string) => void) | null = null;
  private pipelines: Map<string, GPUComputePipeline> = new Map();
  private buffers: Map<string, GPUBuffer> = new Map();
  private bindGroups: Map<string, GPUBindGroup> = new Map();

  private constructor() {}

  /** Initialize WebGPU device. Must be called before any GPU operations. */
  static async create(options: GPUContextOptions = {}): Promise<GPUContext> {
    const ctx = new GPUContext();

    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error(
        'WebGPU is not supported in this environment. ' +
        'Use a WebGPU-capable browser (Chrome 113+, Firefox Nightly, Safari 18+).'
      );
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: options.powerPreference ?? 'high-performance',
    });
    if (!adapter) {
      throw new Error('No WebGPU adapter available.');
    }
    ctx.adapter = adapter;

    const device = await adapter.requestDevice({
      requiredFeatures: options.requiredFeatures ?? [],
    });
    ctx.device = device;

    device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
      ctx.onDeviceLost?.(info.message);
    });

    return ctx;
  }

  /** Create a storage buffer (for body states, constraints, etc.) */
  createStorageBuffer(name: string, sizeBytes: number, usage?: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(sizeBytes, 4), // Minimum 4 bytes
      usage: usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
      mappedAtCreation: false,
    });
    this.buffers.set(name, buffer);
    return buffer;
  }

  /** Create a uniform buffer (for solver params) */
  createUniformBuffer(name: string, sizeBytes: number): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: sizeBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
    });
    this.buffers.set(name, buffer);
    return buffer;
  }

  /** Upload data to a GPU buffer */
  writeBuffer(buffer: GPUBuffer, data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer> | Int32Array<ArrayBuffer>, offset: number = 0): void {
    this.device.queue.writeBuffer(buffer, offset, data);
  }

  /** Read data back from a GPU buffer (async, requires staging buffer) */
  async readBuffer(buffer: GPUBuffer, sizeBytes: number): Promise<ArrayBuffer> {
    const stagingBuffer = this.device.createBuffer({
      size: sizeBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, sizeBytes);
    this.device.queue.submit([encoder.finish()]);

    try {
      await stagingBuffer.mapAsync(GPUMapMode.READ);
    } catch (e) {
      stagingBuffer.destroy();
      throw new Error(`GPU readBuffer failed (device lost?): ${(e as Error).message}`);
    }
    const result = stagingBuffer.getMappedRange().slice(0);
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return result;
  }

  /** Create a compute pipeline from WGSL shader code */
  createComputePipeline(
    name: string,
    shaderCode: string,
    bindGroupLayout: GPUBindGroupLayout,
    entryPoint: string = 'main',
  ): GPUComputePipeline {
    const shaderModule = this.device.createShaderModule({ code: shaderCode });
    const pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint,
      },
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }

  /** Get a named buffer */
  getBuffer(name: string): GPUBuffer | undefined {
    return this.buffers.get(name);
  }

  /** Get a named pipeline */
  getPipeline(name: string): GPUComputePipeline | undefined {
    return this.pipelines.get(name);
  }

  /** Dispatch a compute pass */
  dispatchCompute(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroupCountX: number,
    workgroupCountY: number = 1,
    workgroupCountZ: number = 1,
  ): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
    pass.end();
  }

  /** Clean up all GPU resources */
  destroy(): void {
    for (const buffer of this.buffers.values()) {
      buffer.destroy();
    }
    this.buffers.clear();
    this.pipelines.clear();
    this.bindGroups.clear();
    this.device.destroy();
  }
}
