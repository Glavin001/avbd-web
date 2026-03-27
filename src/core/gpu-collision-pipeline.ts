/**
 * GPU Collision Pipeline — orchestrates the full GPU collision detection workflow.
 *
 * This module provides a high-level API that combines:
 *   1. LBVH broad phase (Morton codes → radix sort → BVH build → traverse)
 *   2. GPU narrow phase (per-shape-pair contact generation)
 *   3. GPU constraint assembly (contacts → solver constraint rows)
 *
 * It encapsulates all GPU buffer management and provides a simple interface
 * for the GPU solver to use.
 */

import { GPUContext } from './gpu-context.js';
import { GpuBvh, type BvhMode } from './gpu-bvh.js';
import { GpuNarrowphase } from './gpu-narrowphase.js';
import type { Body3D } from './rigid-body-3d.js';
import type { Body2D } from './rigid-body.js';
import { ColliderShapeType3D } from './rigid-body-3d.js';
import { ColliderShapeType } from './rigid-body.js';
import { RigidBodyType } from './types.js';

const WG_SIZE = 256;

/** Collider info stride: 8 u32s per body */
const COLLIDER_INFO_STRIDE = 8;

/**
 * High-level GPU collision pipeline for both 2D and 3D.
 */
export class GpuCollisionPipeline {
  private ctx: GPUContext;
  private mode: BvhMode;
  private bvh: GpuBvh;
  private narrowphase: GpuNarrowphase;

  // Body data buffers managed by this pipeline
  private colliderInfoBuf!: GPUBuffer;
  private bodyTypesBuf!: GPUBuffer;
  private capacity = 0;

  // Pre-allocated upload arrays
  private _colliderInfoUpload: Uint32Array = new Uint32Array(0);
  private _bodyTypesUpload: Uint32Array = new Uint32Array(0);

  constructor(ctx: GPUContext, mode: BvhMode) {
    this.ctx = ctx;
    this.mode = mode;
    this.bvh = new GpuBvh(ctx, mode);
    this.narrowphase = new GpuNarrowphase(ctx, mode);
  }

  private ensureCapacity(n: number): void {
    if (n <= this.capacity) return;
    const cap = Math.max(n, 64);
    const device = this.ctx.device;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    if (this.colliderInfoBuf) {
      this.colliderInfoBuf.destroy();
      this.bodyTypesBuf.destroy();
    }

    this.colliderInfoBuf = device.createBuffer({ size: cap * COLLIDER_INFO_STRIDE * 4, usage: storage });
    this.bodyTypesBuf = device.createBuffer({ size: cap * 4, usage: storage });

    this._colliderInfoUpload = new Uint32Array(cap * COLLIDER_INFO_STRIDE);
    this._bodyTypesUpload = new Uint32Array(cap);
    this.capacity = cap;
  }

  /**
   * Upload collider info and body types for 3D bodies.
   * Must be called before runBroadphase.
   */
  uploadBodyData3D(bodies: Body3D[], bodyStateBuf: GPUBuffer): void {
    const n = bodies.length;
    this.ensureCapacity(n);

    const ci = this._colliderInfoUpload;
    const bt = this._bodyTypesUpload;
    const fv = new Float32Array(ci.buffer);

    for (let i = 0; i < n; i++) {
      const b = bodies[i];
      const base = i * COLLIDER_INFO_STRIDE;
      ci[base] = b.colliderShape === ColliderShapeType3D.Ball ? 1 : 0;
      fv[base + 1] = b.halfExtents.x;
      fv[base + 2] = b.halfExtents.y;
      fv[base + 3] = b.halfExtents.z;
      fv[base + 4] = b.radius;
      fv[base + 5] = b.friction;
      fv[base + 6] = b.restitution;
      ci[base + 7] = b.type;
      bt[i] = b.type;
    }

    const device = this.ctx.device;
    (device.queue as any).writeBuffer(this.colliderInfoBuf, 0, ci, 0, n * COLLIDER_INFO_STRIDE);
    (device.queue as any).writeBuffer(this.bodyTypesBuf, 0, bt, 0, n);
  }

  /**
   * Upload collider info and body types for 2D bodies.
   */
  uploadBodyData2D(bodies: Body2D[], bodyStateBuf: GPUBuffer): void {
    const n = bodies.length;
    this.ensureCapacity(n);

    const ci = this._colliderInfoUpload;
    const bt = this._bodyTypesUpload;
    const fv = new Float32Array(ci.buffer);

    for (let i = 0; i < n; i++) {
      const b = bodies[i];
      const base = i * COLLIDER_INFO_STRIDE;
      // Shape type: 0=Cuboid, 1=Ball (matching ColliderShape enum)
      ci[base] = b.colliderShape === ColliderShapeType.Ball ? 1 : 0;
      fv[base + 1] = b.halfExtents.x;
      fv[base + 2] = b.halfExtents.y;
      fv[base + 3] = 0; // no z extent in 2D
      fv[base + 4] = b.radius;
      fv[base + 5] = b.friction;
      fv[base + 6] = b.restitution;
      ci[base + 7] = b.type;
      bt[i] = b.type;
    }

    const device = this.ctx.device;
    (device.queue as any).writeBuffer(this.colliderInfoBuf, 0, ci, 0, n * COLLIDER_INFO_STRIDE);
    (device.queue as any).writeBuffer(this.bodyTypesBuf, 0, bt, 0, n);
  }

  /**
   * Compute scene bounds from body positions (CPU pre-pass).
   * Returns [minX, minY, minZ, maxX, maxY, maxZ] for 3D
   * or [minX, minY, maxX, maxY] for 2D.
   */
  computeSceneBounds3D(bodies: Body3D[]): Float32Array {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const b of bodies) {
      const r = b.boundingRadius;
      minX = Math.min(minX, b.position.x - r);
      minY = Math.min(minY, b.position.y - r);
      minZ = Math.min(minZ, b.position.z - r);
      maxX = Math.max(maxX, b.position.x + r);
      maxY = Math.max(maxY, b.position.y + r);
      maxZ = Math.max(maxZ, b.position.z + r);
    }

    // Add small margin to avoid degenerate scene bounds
    const margin = 0.1;
    return new Float32Array([
      minX - margin, minY - margin, minZ - margin,
      maxX + margin, maxY + margin, maxZ + margin,
    ]);
  }

  computeSceneBounds2D(bodies: Body2D[]): Float32Array {
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const b of bodies) {
      const r = b.boundingRadius;
      minX = Math.min(minX, b.position.x - r);
      minY = Math.min(minY, b.position.y - r);
      maxX = Math.max(maxX, b.position.x + r);
      maxY = Math.max(maxY, b.position.y + r);
    }

    const margin = 0.1;
    return new Float32Array([minX - margin, minY - margin, maxX + margin, maxY + margin]);
  }

  /**
   * Run the full GPU collision pipeline and get the constraint buffer.
   *
   * @returns An object containing the constraint buffer, its count staging buffer,
   * and the pair buffer/count for graph coloring.
   */
  async runFullPipeline(
    n: number,
    bodyStateBuf: GPUBuffer,
    sceneBounds: Float32Array,
    collisionMargin: number,
    dt: number,
    penaltyMin: number,
    alpha: number,
  ): Promise<{
    constraintBuffer: GPUBuffer;
    numConstraints: number;
    pairBuffer: GPUBuffer;
    numPairs: number;
  }> {
    const device = this.ctx.device;
    const encoder = device.createCommandEncoder();

    // 1. BVH broad phase
    const bvhResult = this.bvh.buildAndTraverse(
      encoder, n,
      bodyStateBuf,
      this.colliderInfoBuf,
      this.bodyTypesBuf,
      sceneBounds,
    );

    // Submit BVH pass and read back pair count
    device.queue.submit([encoder.finish()]);
    const numPairs = await this.bvh.readPairCount();

    if (numPairs === 0) {
      return {
        constraintBuffer: this.narrowphase.getConstraintBuffer(),
        numConstraints: 0,
        pairBuffer: bvhResult.pairBuffer,
        numPairs: 0,
      };
    }

    // 2. Narrow phase + constraint assembly
    const encoder2 = device.createCommandEncoder();
    const npResult = this.narrowphase.generateConstraints(
      encoder2,
      numPairs,
      bvhResult.pairBuffer,
      bodyStateBuf,
      this.colliderInfoBuf,
      collisionMargin,
      dt,
      penaltyMin,
      alpha,
    );

    device.queue.submit([encoder2.finish()]);
    const numConstraints = await this.narrowphase.readConstraintCount();

    return {
      constraintBuffer: npResult.constraintBuffer,
      numConstraints,
      pairBuffer: bvhResult.pairBuffer,
      numPairs,
    };
  }

  /** Get direct access to component modules for advanced usage. */
  getBvh(): GpuBvh { return this.bvh; }
  getNarrowphase(): GpuNarrowphase { return this.narrowphase; }
  getColliderInfoBuffer(): GPUBuffer { return this.colliderInfoBuf; }
  getBodyTypesBuffer(): GPUBuffer { return this.bodyTypesBuf; }

  destroy(): void {
    this.bvh.destroy();
    this.narrowphase.destroy();
    this.colliderInfoBuf?.destroy();
    this.bodyTypesBuf?.destroy();
  }
}
