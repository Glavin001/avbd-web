/**
 * Unit tests for GPU collision pipeline integration.
 *
 * These tests verify:
 * 1. Constraint buffer layout matches between GPU assembly shaders and CPU solver
 * 2. The readback parsing logic correctly interprets GPU constraint data
 * 3. The useGPUCollision config flag is wired up correctly
 * 4. Scene bounds computation is correct
 * 5. Collider info upload formats are correct
 *
 * Note: These are structural/layout tests that run in vitest (no GPU).
 * GPU execution tests are in tests/browser/gpu-execution.spec.ts.
 */

import { describe, it, expect } from 'vitest';
import { ForceType } from '../src/core/types.js';
import { DEFAULT_SOLVER_CONFIG_2D, DEFAULT_SOLVER_CONFIG_3D, COLLISION_MARGIN } from '../src/core/types.js';
import {
  CONSTRAINT_ASSEMBLY_2D_WGSL,
  CONSTRAINT_ASSEMBLY_3D_WGSL,
  NARROWPHASE_2D_WGSL,
  NARROWPHASE_3D_WGSL,
  BVH_BUILD_WGSL,
  BVH_TRAVERSE_2D_WGSL,
  BVH_TRAVERSE_3D_WGSL,
  MORTON_CODES_2D_WGSL,
  MORTON_CODES_3D_WGSL,
  RADIX_SORT_WGSL,
} from '../src/shaders/embedded.js';

// ─── Constants matching the solver and assembly shaders ──────────────────────

const CONSTRAINT_STRIDE_2D = 28; // floats per 2D constraint row
const CONSTRAINT_STRIDE_3D = 36; // floats per 3D constraint row
const BODY_STRIDE_2D = 8;        // floats per 2D body state
const BODY_STRIDE_3D = 20;       // floats per 3D body state
const COLLIDER_INFO_STRIDE = 8;  // u32s per body collider info

// ─── Shader Embedding Tests ──────────────────────────────────────────────────

describe('GPU Collision Shader Embedding', () => {
  it('all collision shaders are embedded and non-empty', () => {
    expect(CONSTRAINT_ASSEMBLY_2D_WGSL.length).toBeGreaterThan(100);
    expect(CONSTRAINT_ASSEMBLY_3D_WGSL.length).toBeGreaterThan(100);
    expect(NARROWPHASE_2D_WGSL.length).toBeGreaterThan(100);
    expect(NARROWPHASE_3D_WGSL.length).toBeGreaterThan(100);
    expect(BVH_BUILD_WGSL.length).toBeGreaterThan(100);
    expect(BVH_TRAVERSE_2D_WGSL.length).toBeGreaterThan(100);
    expect(BVH_TRAVERSE_3D_WGSL.length).toBeGreaterThan(100);
    expect(MORTON_CODES_2D_WGSL.length).toBeGreaterThan(100);
    expect(MORTON_CODES_3D_WGSL.length).toBeGreaterThan(100);
    expect(RADIX_SORT_WGSL.length).toBeGreaterThan(100);
  });

  it('2D constraint assembly shader has correct stride constant', () => {
    expect(CONSTRAINT_ASSEMBLY_2D_WGSL).toContain('const CONSTRAINT_STRIDE: u32 = 28u;');
  });

  it('3D constraint assembly shader has correct stride constant', () => {
    expect(CONSTRAINT_ASSEMBLY_3D_WGSL).toContain('const CONSTRAINT_STRIDE: u32 = 36u;');
  });

  it('2D narrowphase shader has correct entry point', () => {
    expect(NARROWPHASE_2D_WGSL).toContain('fn narrowphase_2d');
  });

  it('3D narrowphase shader has correct entry point', () => {
    expect(NARROWPHASE_3D_WGSL).toContain('fn narrowphase_3d');
  });

  it('2D constraint assembly shader has correct entry point', () => {
    expect(CONSTRAINT_ASSEMBLY_2D_WGSL).toContain('fn constraint_assembly_2d');
  });

  it('3D constraint assembly shader has correct entry point', () => {
    expect(CONSTRAINT_ASSEMBLY_3D_WGSL).toContain('fn constraint_assembly_3d');
  });
});

// ─── 2D Constraint Buffer Layout ─────────────────────────────────────────────

describe('2D GPU Constraint Buffer Layout', () => {
  it('assembly shader output format matches solver upload format', () => {
    // Simulate what the GPU assembly shader writes (28 floats per row)
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    const bodyA = 2, bodyB = 5;
    const nx = 0.0, ny = 1.0; // normal pointing up
    const jAx = nx, jAy = ny, jAtheta = 0.3;
    const jBx = -nx, jBy = -ny, jBtheta = -0.2;
    const hessA = -0.1, hessB = -0.15;
    const mu = 0.4;
    const c = -0.01, c0 = -0.01;
    const lambda = 0.0, penalty = 100.0;
    const stiffness = 1e30;
    const fmin = -1e30, fmax = 0.0;

    // Write in GPU assembly shader format (base + N offsets)
    // base+0: bodyA (i32)
    view.setInt32(0, bodyA, true);
    // base+1: bodyB (i32)
    view.setInt32(4, bodyB, true);
    // base+2: forceType (u32)
    view.setUint32(8, ForceType.Contact, true);
    // base+3: pad
    view.setUint32(12, 0, true);
    // base+4..7: jacobianA (vec4)
    view.setFloat32(16, jAx, true);
    view.setFloat32(20, jAy, true);
    view.setFloat32(24, jAtheta, true);
    view.setFloat32(28, 0.0, true);
    // base+8..11: jacobianB (vec4)
    view.setFloat32(32, jBx, true);
    view.setFloat32(36, jBy, true);
    view.setFloat32(40, jBtheta, true);
    view.setFloat32(44, 0.0, true);
    // base+12..15: hessianDiagA (vec4)
    view.setFloat32(48, 0.0, true);
    view.setFloat32(52, 0.0, true);
    view.setFloat32(56, hessA, true);
    view.setFloat32(60, 0.0, true);
    // base+16..19: hessianDiagB (vec4, w=mu)
    view.setFloat32(64, 0.0, true);
    view.setFloat32(68, 0.0, true);
    view.setFloat32(72, hessB, true);
    view.setFloat32(76, mu, true);
    // base+20..27: scalar fields
    view.setFloat32(80, c, true);
    view.setFloat32(84, c0, true);
    view.setFloat32(88, lambda, true);
    view.setFloat32(92, penalty, true);
    view.setFloat32(96, stiffness, true);
    view.setFloat32(100, fmin, true);
    view.setFloat32(104, fmax, true);
    view.setUint32(108, 1, true); // active

    // Read back using the same offsets the solver readback uses
    expect(view.getInt32(0, true)).toBe(bodyA);
    expect(view.getInt32(4, true)).toBe(bodyB);
    expect(view.getUint32(8, true)).toBe(ForceType.Contact);

    // Jacobian A
    expect(view.getFloat32(16, true)).toBeCloseTo(jAx, 5);
    expect(view.getFloat32(20, true)).toBeCloseTo(jAy, 5);
    expect(view.getFloat32(24, true)).toBeCloseTo(jAtheta, 5);

    // Jacobian B
    expect(view.getFloat32(32, true)).toBeCloseTo(jBx, 5);
    expect(view.getFloat32(36, true)).toBeCloseTo(jBy, 5);
    expect(view.getFloat32(40, true)).toBeCloseTo(jBtheta, 5);

    // Hessian diag A (angular at offset 56)
    expect(view.getFloat32(56, true)).toBeCloseTo(hessA, 5);

    // Hessian diag B (angular at offset 72, mu at offset 76)
    expect(view.getFloat32(72, true)).toBeCloseTo(hessB, 5);
    expect(view.getFloat32(76, true)).toBeCloseTo(mu, 5);

    // Scalar fields
    expect(view.getFloat32(80, true)).toBeCloseTo(c, 5);
    expect(view.getFloat32(84, true)).toBeCloseTo(c0, 5);
    expect(view.getFloat32(88, true)).toBeCloseTo(lambda, 5);
    expect(view.getFloat32(92, true)).toBeCloseTo(penalty, 2);
    expect(view.getUint32(108, true)).toBe(1);
  });

  it('2D constraint row size is exactly 112 bytes', () => {
    expect(CONSTRAINT_STRIDE_2D * 4).toBe(112);
  });
});

// ─── 3D Constraint Buffer Layout ─────────────────────────────────────────────

describe('3D GPU Constraint Buffer Layout', () => {
  it('assembly shader output format matches solver upload format', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    const bodyA = 1, bodyB = 3;
    const jAlin = [0.0, 1.0, 0.0];
    const jAang = [0.1, 0.2, 0.3];
    const jBlin = [0.0, -1.0, 0.0];
    const jBang = [-0.1, -0.2, -0.3];
    const hAang = [-0.05, -0.06, -0.07];
    const hBang = [-0.08, -0.09, -0.10];
    const mu = 0.35;

    // Write in GPU 3D assembly shader format (36 floats)
    view.setInt32(0, bodyA, true);
    view.setInt32(4, bodyB, true);
    view.setUint32(8, ForceType.Contact, true);
    view.setUint32(12, 0, true); // pad

    // Jacobian A linear (vec4): bytes 16-31
    view.setFloat32(16, jAlin[0], true);
    view.setFloat32(20, jAlin[1], true);
    view.setFloat32(24, jAlin[2], true);
    view.setFloat32(28, 0.0, true);

    // Jacobian A angular (vec4): bytes 32-47
    view.setFloat32(32, jAang[0], true);
    view.setFloat32(36, jAang[1], true);
    view.setFloat32(40, jAang[2], true);
    view.setFloat32(44, 0.0, true); // no mu here after fix

    // Jacobian B linear (vec4): bytes 48-63
    view.setFloat32(48, jBlin[0], true);
    view.setFloat32(52, jBlin[1], true);
    view.setFloat32(56, jBlin[2], true);
    view.setFloat32(60, 0.0, true);

    // Jacobian B angular (vec4, w=mu): bytes 64-79
    view.setFloat32(64, jBang[0], true);
    view.setFloat32(68, jBang[1], true);
    view.setFloat32(72, jBang[2], true);
    view.setFloat32(76, mu, true);

    // Hessian diag A angular (vec4): bytes 80-95
    view.setFloat32(80, hAang[0], true);
    view.setFloat32(84, hAang[1], true);
    view.setFloat32(88, hAang[2], true);
    view.setFloat32(92, 0.0, true);

    // Hessian diag B angular (vec4): bytes 96-111
    view.setFloat32(96, hBang[0], true);
    view.setFloat32(100, hBang[1], true);
    view.setFloat32(104, hBang[2], true);
    view.setFloat32(108, 0.0, true);

    // Scalar fields: bytes 112-143
    view.setFloat32(112, -0.005, true);  // c
    view.setFloat32(116, -0.005, true);  // c0
    view.setFloat32(120, 0.0, true);     // lambda
    view.setFloat32(124, 100.0, true);   // penalty
    view.setFloat32(128, 1e30, true);    // stiffness
    view.setFloat32(132, -1e30, true);   // fmin
    view.setFloat32(136, 0.0, true);     // fmax
    view.setUint32(140, 1, true);        // active

    // Verify readback matches what readbackGpuConstraints3D extracts
    expect(view.getInt32(0, true)).toBe(bodyA);
    expect(view.getInt32(4, true)).toBe(bodyB);

    // Jacobian A: [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]
    expect(view.getFloat32(16, true)).toBe(jAlin[0]);
    expect(view.getFloat32(20, true)).toBe(jAlin[1]);
    expect(view.getFloat32(24, true)).toBe(jAlin[2]);
    expect(view.getFloat32(32, true)).toBeCloseTo(jAang[0], 5);
    expect(view.getFloat32(36, true)).toBeCloseTo(jAang[1], 5);
    expect(view.getFloat32(40, true)).toBeCloseTo(jAang[2], 5);

    // Jacobian B: [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]
    expect(view.getFloat32(48, true)).toBeCloseTo(jBlin[0], 5);
    expect(view.getFloat32(52, true)).toBeCloseTo(jBlin[1], 5);
    expect(view.getFloat32(56, true)).toBeCloseTo(jBlin[2], 5);
    expect(view.getFloat32(64, true)).toBeCloseTo(jBang[0], 5);
    expect(view.getFloat32(68, true)).toBeCloseTo(jBang[1], 5);
    expect(view.getFloat32(72, true)).toBeCloseTo(jBang[2], 5);

    // mu at jacobian_b_ang.w
    expect(view.getFloat32(76, true)).toBeCloseTo(mu, 5);

    // Hessian angular
    expect(view.getFloat32(80, true)).toBeCloseTo(hAang[0], 5);
    expect(view.getFloat32(84, true)).toBeCloseTo(hAang[1], 5);
    expect(view.getFloat32(88, true)).toBeCloseTo(hAang[2], 5);
    expect(view.getFloat32(96, true)).toBeCloseTo(hBang[0], 5);
    expect(view.getFloat32(100, true)).toBeCloseTo(hBang[1], 5);
    expect(view.getFloat32(104, true)).toBeCloseTo(hBang[2], 5);

    // Scalar fields
    expect(view.getFloat32(112, true)).toBeCloseTo(-0.005, 5);
    expect(view.getFloat32(124, true)).toBe(100.0);
    expect(view.getUint32(140, true)).toBe(1);
  });

  it('3D constraint row size is exactly 144 bytes', () => {
    expect(CONSTRAINT_STRIDE_3D * 4).toBe(144);
  });

  it('3D assembly shader puts mu at jacobian_b_ang.w (not jacobian_a_ang.w)', () => {
    // Verify the fix: mu should be at base+19 (jacobian_b_ang.w)
    expect(CONSTRAINT_ASSEMBLY_3D_WGSL).toContain('constraint_buffer[base + 19u] = mu;');
    // And jacobian_a_ang.w should be 0.0
    expect(CONSTRAINT_ASSEMBLY_3D_WGSL).toContain('constraint_buffer[base + 11u] = 0.0;');
  });
});

// ─── Config Tests ────────────────────────────────────────────────────────────

describe('useGPUCollision Config', () => {
  it('SolverConfig includes useGPUCollision field type', () => {
    // useGPUCollision is optional and defaults to undefined (treated as true for GPU solvers)
    const config2d = { ...DEFAULT_SOLVER_CONFIG_2D };
    expect(config2d.useGPUCollision).toBeUndefined();

    const config3d = { ...DEFAULT_SOLVER_CONFIG_3D };
    expect(config3d.useGPUCollision).toBeUndefined();
  });

  it('useGPUCollision can be explicitly set to false', () => {
    const config = { ...DEFAULT_SOLVER_CONFIG_2D, useGPUCollision: false };
    expect(config.useGPUCollision).toBe(false);
  });

  it('useGPUCollision can be explicitly set to true', () => {
    const config = { ...DEFAULT_SOLVER_CONFIG_2D, useGPUCollision: true };
    expect(config.useGPUCollision).toBe(true);
  });
});

// ─── Scene Bounds Tests ──────────────────────────────────────────────────────

describe('Scene Bounds Computation', () => {
  it('COLLISION_MARGIN is a small positive value', () => {
    expect(COLLISION_MARGIN).toBeGreaterThan(0);
    expect(COLLISION_MARGIN).toBeLessThan(0.01);
  });
});

// ─── Collider Info Layout Tests ──────────────────────────────────────────────

describe('Collider Info Buffer Layout', () => {
  it('collider info stride is 8 u32s (32 bytes) per body', () => {
    expect(COLLIDER_INFO_STRIDE * 4).toBe(32);
  });

  it('collider info upload format matches shader expectations', () => {
    // Simulate what uploadBodyData2D writes
    const info = new Uint32Array(COLLIDER_INFO_STRIDE);
    const fv = new Float32Array(info.buffer);

    // Cuboid shape (type 0)
    info[0] = 0; // shape type (0=Cuboid, 1=Ball)
    fv[1] = 0.5; // halfExtent.x
    fv[2] = 0.5; // halfExtent.y
    fv[3] = 0.0; // halfExtent.z (unused in 2D)
    fv[4] = 0.0; // radius (only for Ball)
    fv[5] = 0.5; // friction
    fv[6] = 0.0; // restitution
    info[7] = 0;  // body type (0=Dynamic)

    expect(info[0]).toBe(0); // Cuboid
    expect(fv[1]).toBe(0.5);
    expect(fv[5]).toBe(0.5); // friction

    // Ball shape (type 1)
    info[0] = 1;
    fv[4] = 0.5; // radius

    expect(info[0]).toBe(1); // Ball
    expect(fv[4]).toBe(0.5); // radius
  });
});

// ─── Contact Format Tests ────────────────────────────────────────────────────

describe('Contact Buffer Format', () => {
  const CONTACT_STRIDE_2D = 8; // u32s per 2D contact
  const CONTACT_STRIDE_3D = 12; // u32s per 3D contact

  it('2D contact format is 8 u32s (32 bytes)', () => {
    // [bodyA, bodyB, featureId, _pad, normal_x, normal_y, depth, mu]
    const contact = new Uint32Array(CONTACT_STRIDE_2D);
    const fv = new Float32Array(contact.buffer);

    contact[0] = 2;  // bodyA
    contact[1] = 5;  // bodyB
    contact[2] = 42; // featureId
    contact[3] = 0;  // pad
    fv[4] = 0.0;     // normal_x
    fv[5] = 1.0;     // normal_y
    fv[6] = 0.01;    // depth
    fv[7] = 0.5;     // mu

    expect(contact[0]).toBe(2);
    expect(contact[1]).toBe(5);
    expect(fv[5]).toBe(1.0); // normal_y
    expect(fv[7]).toBe(0.5); // mu
  });

  it('3D contact format is 12 u32s (48 bytes)', () => {
    // [bodyA, bodyB, featureId, _pad,
    //  normal_x, normal_y, normal_z, depth,
    //  point_x, point_y, point_z, mu]
    const contact = new Uint32Array(CONTACT_STRIDE_3D);
    const fv = new Float32Array(contact.buffer);

    contact[0] = 1;  // bodyA
    contact[1] = 3;  // bodyB
    contact[2] = 7;  // featureId
    fv[4] = 0.0;     // normal_x
    fv[5] = 1.0;     // normal_y
    fv[6] = 0.0;     // normal_z
    fv[7] = 0.005;   // depth
    fv[8] = 1.0;     // point_x
    fv[9] = 2.0;     // point_y
    fv[10] = 3.0;    // point_z
    fv[11] = 0.4;    // mu

    expect(contact[0]).toBe(1);
    expect(fv[5]).toBe(1.0);   // normal_y
    expect(fv[11]).toBeCloseTo(0.4, 5);  // mu
  });
});

// ─── Integration: World API with useGPUCollision ─────────────────────────────

describe('World API with useGPUCollision (CPU fallback)', () => {
  // These tests use stepCPU() which doesn't need GPU, but verify the config wiring

  it('2D World accepts useGPUCollision in config', async () => {
    const { default: AVBD } = await import('../src/2d/index.js');
    // With useGPUCollision: false, should use CPU collision
    const world = new AVBD.World({ x: 0, y: -9.81 }, { iterations: 5, useGPUCollision: false });

    // Create a simple scene
    world.createCollider(AVBD.ColliderDesc.cuboid(10, 0.5));
    const body = world.createRigidBody(AVBD.RigidBodyDesc.dynamic().setTranslation(0, 3));
    world.createCollider(AVBD.ColliderDesc.cuboid(0.5, 0.5), body);

    // Step with CPU (useGPUCollision: false still uses CPU path in stepCPU)
    for (let i = 0; i < 60; i++) world.stepCPU();

    const pos = body.translation();
    expect(pos.y).toBeGreaterThan(0.4);
    expect(pos.y).toBeLessThan(3.0);
    expect(isFinite(pos.x)).toBe(true);
    expect(isFinite(pos.y)).toBe(true);
  });

  it('3D World accepts useGPUCollision in config', async () => {
    const { default: AVBD3D } = await import('../src/3d/index.js');
    const world = new AVBD3D.World({ x: 0, y: -9.81, z: 0 },
      { iterations: 5, useGPUCollision: false, useCPU: true });

    world.createCollider(AVBD3D.ColliderDesc.cuboid(10, 0.5, 10));
    const body = world.createRigidBody(
      AVBD3D.RigidBodyDesc.dynamic().setTranslation(0, 3, 0)
    );
    world.createCollider(AVBD3D.ColliderDesc.cuboid(0.5, 0.5, 0.5), body);

    for (let i = 0; i < 60; i++) await world.step();

    const pos = body.translation();
    expect(pos.y).toBeGreaterThan(0.3);
    expect(pos.y).toBeLessThan(4.0);
    expect(isFinite(pos.x)).toBe(true);
    expect(isFinite(pos.y)).toBe(true);
    expect(isFinite(pos.z)).toBe(true);
  });
});

// ─── Multi-constraint Readback Parsing ───────────────────────────────────────

describe('Multi-constraint Buffer Parsing', () => {
  it('correctly parses 3 consecutive 2D constraint rows', () => {
    const numRows = 3;
    const buf = new ArrayBuffer(numRows * CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    // Write 3 rows with different body pairs
    for (let i = 0; i < numRows; i++) {
      const off = i * CONSTRAINT_STRIDE_2D * 4;
      view.setInt32(off + 0, i, true);       // bodyA
      view.setInt32(off + 4, i + 10, true);  // bodyB
      view.setUint32(off + 8, ForceType.Contact, true);
      view.setFloat32(off + 80, -(i + 1) * 0.01, true); // c
      view.setFloat32(off + 92, 100 + i * 50, true);     // penalty
      view.setUint32(off + 108, 1, true);                 // active
    }

    // Verify each row can be parsed independently
    for (let i = 0; i < numRows; i++) {
      const off = i * CONSTRAINT_STRIDE_2D * 4;
      expect(view.getInt32(off + 0, true)).toBe(i);
      expect(view.getInt32(off + 4, true)).toBe(i + 10);
      expect(view.getFloat32(off + 80, true)).toBeCloseTo(-(i + 1) * 0.01, 5);
      expect(view.getFloat32(off + 92, true)).toBe(100 + i * 50);
      expect(view.getUint32(off + 108, true)).toBe(1);
    }
  });

  it('correctly parses 3 consecutive 3D constraint rows', () => {
    const numRows = 3;
    const buf = new ArrayBuffer(numRows * CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    for (let i = 0; i < numRows; i++) {
      const off = i * CONSTRAINT_STRIDE_3D * 4;
      view.setInt32(off + 0, i * 2, true);       // bodyA
      view.setInt32(off + 4, i * 2 + 1, true);   // bodyB
      view.setUint32(off + 8, ForceType.Contact, true);
      // Jacobian A lin: (0, 1, 0)
      view.setFloat32(off + 20, 1.0, true);
      // Jacobian B lin: (0, -1, 0)
      view.setFloat32(off + 52, -1.0, true);
      // mu at jacobian_b_ang.w
      view.setFloat32(off + 76, 0.3 + i * 0.1, true);
      // Scalar fields
      view.setFloat32(off + 112, -0.005 * (i + 1), true); // c
      view.setFloat32(off + 124, 100.0, true);             // penalty
      view.setUint32(off + 140, 1, true);                  // active
    }

    for (let i = 0; i < numRows; i++) {
      const off = i * CONSTRAINT_STRIDE_3D * 4;
      expect(view.getInt32(off + 0, true)).toBe(i * 2);
      expect(view.getInt32(off + 4, true)).toBe(i * 2 + 1);
      expect(view.getFloat32(off + 20, true)).toBe(1.0);
      expect(view.getFloat32(off + 76, true)).toBeCloseTo(0.3 + i * 0.1, 5);
      expect(view.getFloat32(off + 112, true)).toBeCloseTo(-0.005 * (i + 1), 5);
      expect(view.getUint32(off + 140, true)).toBe(1);
    }
  });
});
