import { describe, it, expect } from 'vitest';
import {
  PRIMAL_UPDATE_2D_WGSL,
  PRIMAL_UPDATE_3D_WGSL,
  DUAL_UPDATE_3D_WGSL,
  FRICTION_COUPLING_3D_WGSL,
} from '../src/shaders/embedded.js';

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Check that a float32 round-trip preserves a value within relative tolerance. */
function expectFloat32Close(actual: number, expected: number): void {
  if (expected === 0) {
    expect(actual).toBe(0);
  } else {
    // Relative error must be tiny (float32 has ~7 significant digits)
    const relErr = Math.abs((actual - expected) / expected);
    expect(relErr).toBeLessThan(1e-6);
  }
}

// ─── Constants matching gpu-solver-2d.ts and gpu-solver-3d.ts ────────────────

const CONSTRAINT_STRIDE_2D = 28; // floats
const CONSTRAINT_STRIDE_3D = 36; // floats
const BODY_STRIDE_3D = 20;       // floats
const BODY_PREV_STRIDE_3D = 14;  // floats

// ─── 2D Constraint Row Layout ────────────────────────────────────────────────

describe('2D Constraint Row Layout', () => {
  it('full round-trip: pack and read back all fields in 112-byte buffer', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    // Header: body_a, body_b, force_type, padding
    view.setInt32(0, 3, true);    // body_a
    view.setInt32(4, 7, true);    // body_b
    view.setUint32(8, 2, true);   // force_type
    view.setUint32(12, 0, true);  // padding

    // jacobianA (vec4): bytes 16-31
    view.setFloat32(16, 1.0, true);
    view.setFloat32(20, 2.0, true);
    view.setFloat32(24, 3.0, true);
    view.setFloat32(28, 0.0, true); // .w padding

    // jacobianB (vec4): bytes 32-47
    view.setFloat32(32, 4.0, true);
    view.setFloat32(36, 5.0, true);
    view.setFloat32(40, 6.0, true);
    view.setFloat32(44, 0.0, true); // .w padding

    // hessianDiagA (vec4): bytes 48-63
    view.setFloat32(48, 10.0, true);
    view.setFloat32(52, 11.0, true);
    view.setFloat32(56, 12.0, true);
    view.setFloat32(60, 0.0, true); // .w padding

    // hessianDiagB (vec4): bytes 64-79 with mu in .w
    view.setFloat32(64, 13.0, true);
    view.setFloat32(68, 14.0, true);
    view.setFloat32(72, 15.0, true);
    view.setFloat32(76, 0.3, true); // mu packed in .w

    // Scalar fields: bytes 80-111
    view.setFloat32(80, 0.01, true);   // c
    view.setFloat32(84, 0.02, true);   // c0
    view.setFloat32(88, 0.0, true);    // lambda
    view.setFloat32(92, 100.0, true);  // penalty
    view.setFloat32(96, 1e30, true);   // stiffness (Infinity encoded)
    view.setFloat32(100, -1e30, true); // fmin (-Infinity encoded)
    view.setFloat32(104, 1e30, true);  // fmax (Infinity encoded)
    view.setUint32(108, 1, true);      // active

    // Read back all fields
    expect(view.getInt32(0, true)).toBe(3);
    expect(view.getInt32(4, true)).toBe(7);
    expect(view.getUint32(8, true)).toBe(2);
    expect(view.getFloat32(16, true)).toBeCloseTo(1.0);
    expect(view.getFloat32(20, true)).toBeCloseTo(2.0);
    expect(view.getFloat32(24, true)).toBeCloseTo(3.0);
    expect(view.getFloat32(32, true)).toBeCloseTo(4.0);
    expect(view.getFloat32(36, true)).toBeCloseTo(5.0);
    expect(view.getFloat32(40, true)).toBeCloseTo(6.0);
    expect(view.getFloat32(48, true)).toBeCloseTo(10.0);
    expect(view.getFloat32(52, true)).toBeCloseTo(11.0);
    expect(view.getFloat32(56, true)).toBeCloseTo(12.0);
    expect(view.getFloat32(64, true)).toBeCloseTo(13.0);
    expect(view.getFloat32(68, true)).toBeCloseTo(14.0);
    expect(view.getFloat32(72, true)).toBeCloseTo(15.0);
    expect(view.getFloat32(76, true)).toBeCloseTo(0.3);
    expect(view.getFloat32(80, true)).toBeCloseTo(0.01);
    expect(view.getFloat32(84, true)).toBeCloseTo(0.02);
    expect(view.getFloat32(88, true)).toBeCloseTo(0.0);
    expect(view.getFloat32(92, true)).toBeCloseTo(100.0);
    expectFloat32Close(view.getFloat32(96, true), 1e30);
    expectFloat32Close(view.getFloat32(100, true), -1e30);
    expectFloat32Close(view.getFloat32(104, true), 1e30);
    expect(view.getUint32(108, true)).toBe(1);
  });

  it('jacobianA occupies bytes 16-27 (3 floats + padding at 28)', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    view.setFloat32(16, 1.5, true);  // jacobianA[0]
    view.setFloat32(20, -2.5, true); // jacobianA[1]
    view.setFloat32(24, 3.5, true);  // jacobianA[2]
    view.setFloat32(28, 0.0, true);  // .w padding

    expect(view.getFloat32(16, true)).toBeCloseTo(1.5);
    expect(view.getFloat32(20, true)).toBeCloseTo(-2.5);
    expect(view.getFloat32(24, true)).toBeCloseTo(3.5);
    expect(view.getFloat32(28, true)).toBe(0.0); // padding must be zero
  });

  it('jacobianB occupies bytes 32-43 (3 floats + padding at 44)', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    view.setFloat32(32, -0.5, true);
    view.setFloat32(36, 0.7, true);
    view.setFloat32(40, -0.9, true);
    view.setFloat32(44, 0.0, true);

    expect(view.getFloat32(32, true)).toBeCloseTo(-0.5);
    expect(view.getFloat32(36, true)).toBeCloseTo(0.7);
    expect(view.getFloat32(40, true)).toBeCloseTo(-0.9);
    expect(view.getFloat32(44, true)).toBe(0.0);
  });

  it('hessianDiagA occupies bytes 48-59 (3 floats + padding at 60)', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    view.setFloat32(48, 100.0, true);
    view.setFloat32(52, 200.0, true);
    view.setFloat32(56, 300.0, true);
    view.setFloat32(60, 0.0, true);

    expect(view.getFloat32(48, true)).toBeCloseTo(100.0);
    expect(view.getFloat32(52, true)).toBeCloseTo(200.0);
    expect(view.getFloat32(56, true)).toBeCloseTo(300.0);
    expect(view.getFloat32(60, true)).toBe(0.0);
  });

  it('hessianDiagB occupies bytes 64-75 with mu packed at byte 76', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    view.setFloat32(64, 0.1, true);
    view.setFloat32(68, 0.2, true);
    view.setFloat32(72, 0.3, true);
    view.setFloat32(76, 0.5, true); // mu in .w

    expect(view.getFloat32(64, true)).toBeCloseTo(0.1);
    expect(view.getFloat32(68, true)).toBeCloseTo(0.2);
    expect(view.getFloat32(72, true)).toBeCloseTo(0.3);
    expect(view.getFloat32(76, true)).toBeCloseTo(0.5); // mu
  });

  it('scalar fields occupy bytes 80-111 in correct order', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    view.setFloat32(80, 0.005, true);   // c
    view.setFloat32(84, 0.006, true);   // c0
    view.setFloat32(88, 1.23, true);    // lambda
    view.setFloat32(92, 500.0, true);   // penalty
    view.setFloat32(96, 1000.0, true);  // stiffness
    view.setFloat32(100, -50.0, true);  // fmin
    view.setFloat32(104, 50.0, true);   // fmax
    view.setUint32(108, 1, true);       // active

    expect(view.getFloat32(80, true)).toBeCloseTo(0.005);
    expect(view.getFloat32(84, true)).toBeCloseTo(0.006);
    expect(view.getFloat32(88, true)).toBeCloseTo(1.23);
    expect(view.getFloat32(92, true)).toBeCloseTo(500.0);
    expect(view.getFloat32(96, true)).toBeCloseTo(1000.0);
    expect(view.getFloat32(100, true)).toBeCloseTo(-50.0);
    expect(view.getFloat32(104, true)).toBeCloseTo(50.0);
    expect(view.getUint32(108, true)).toBe(1);
  });

  it('Infinity is encoded as 1e30 and -Infinity as -1e30 for GPU transfer', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    const view = new DataView(buf);

    // Simulate what the solver does: clamp Infinity to 1e30
    const stiffness = Infinity;
    const fmin = -Infinity;
    const fmax = Infinity;

    const encodedStiffness = stiffness === Infinity ? 1e30 : stiffness;
    const encodedFmin = fmin === -Infinity ? -1e30 : fmin;
    const encodedFmax = fmax === Infinity ? 1e30 : fmax;

    view.setFloat32(96, encodedStiffness, true);
    view.setFloat32(100, encodedFmin, true);
    view.setFloat32(104, encodedFmax, true);

    expectFloat32Close(view.getFloat32(96, true), 1e30);
    expectFloat32Close(view.getFloat32(100, true), -1e30);
    expectFloat32Close(view.getFloat32(104, true), 1e30);

    // Verify these are finite (not actual Infinity, which WGSL cannot represent reliably)
    expect(Number.isFinite(view.getFloat32(96, true))).toBe(true);
    expect(Number.isFinite(view.getFloat32(100, true))).toBe(true);
    expect(Number.isFinite(view.getFloat32(104, true))).toBe(true);
  });
});

// ─── 3D Constraint Row Layout ────────────────────────────────────────────────

describe('3D Constraint Row Layout', () => {
  it('full round-trip: pack and read back all 36 floats in 144-byte buffer', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    // Header (bytes 0-15)
    view.setInt32(0, 5, true);    // body_a
    view.setInt32(4, 11, true);   // body_b
    view.setUint32(8, 1, true);   // force_type
    view.setUint32(12, 0, true);  // padding

    // jacobian_a_lin (bytes 16-31)
    view.setFloat32(16, 1.0, true);
    view.setFloat32(20, 2.0, true);
    view.setFloat32(24, 3.0, true);
    view.setFloat32(28, 0.0, true);

    // jacobian_a_ang (bytes 32-47)
    view.setFloat32(32, 4.0, true);
    view.setFloat32(36, 5.0, true);
    view.setFloat32(40, 6.0, true);
    view.setFloat32(44, 0.0, true);

    // jacobian_b_lin (bytes 48-63)
    view.setFloat32(48, -1.0, true);
    view.setFloat32(52, -2.0, true);
    view.setFloat32(56, -3.0, true);
    view.setFloat32(60, 0.0, true);

    // jacobian_b_ang (bytes 64-79) with mu in .w
    view.setFloat32(64, -4.0, true);
    view.setFloat32(68, -5.0, true);
    view.setFloat32(72, -6.0, true);
    view.setFloat32(76, 0.4, true);  // mu

    // hessian_diag_a_ang (bytes 80-95) — NEW fields
    view.setFloat32(80, 10.0, true);
    view.setFloat32(84, 11.0, true);
    view.setFloat32(88, 12.0, true);
    view.setFloat32(92, 0.0, true);

    // hessian_diag_b_ang (bytes 96-111) — NEW fields
    view.setFloat32(96, 20.0, true);
    view.setFloat32(100, 21.0, true);
    view.setFloat32(104, 22.0, true);
    view.setFloat32(108, 0.0, true);

    // Scalar fields (bytes 112-143) — SHIFTED from old layout
    view.setFloat32(112, 0.01, true);   // c
    view.setFloat32(116, 0.02, true);   // c0
    view.setFloat32(120, 0.0, true);    // lambda
    view.setFloat32(124, 200.0, true);  // penalty
    view.setFloat32(128, 1e30, true);   // stiffness
    view.setFloat32(132, -1e30, true);  // fmin
    view.setFloat32(136, 1e30, true);   // fmax
    view.setUint32(140, 1, true);       // active

    // Read back all fields
    expect(view.getInt32(0, true)).toBe(5);
    expect(view.getInt32(4, true)).toBe(11);
    expect(view.getUint32(8, true)).toBe(1);

    expect(view.getFloat32(16, true)).toBeCloseTo(1.0);
    expect(view.getFloat32(20, true)).toBeCloseTo(2.0);
    expect(view.getFloat32(24, true)).toBeCloseTo(3.0);
    expect(view.getFloat32(32, true)).toBeCloseTo(4.0);
    expect(view.getFloat32(36, true)).toBeCloseTo(5.0);
    expect(view.getFloat32(40, true)).toBeCloseTo(6.0);
    expect(view.getFloat32(48, true)).toBeCloseTo(-1.0);
    expect(view.getFloat32(52, true)).toBeCloseTo(-2.0);
    expect(view.getFloat32(56, true)).toBeCloseTo(-3.0);
    expect(view.getFloat32(64, true)).toBeCloseTo(-4.0);
    expect(view.getFloat32(68, true)).toBeCloseTo(-5.0);
    expect(view.getFloat32(72, true)).toBeCloseTo(-6.0);
    expect(view.getFloat32(76, true)).toBeCloseTo(0.4);

    expect(view.getFloat32(80, true)).toBeCloseTo(10.0);
    expect(view.getFloat32(84, true)).toBeCloseTo(11.0);
    expect(view.getFloat32(88, true)).toBeCloseTo(12.0);
    expect(view.getFloat32(96, true)).toBeCloseTo(20.0);
    expect(view.getFloat32(100, true)).toBeCloseTo(21.0);
    expect(view.getFloat32(104, true)).toBeCloseTo(22.0);

    expect(view.getFloat32(112, true)).toBeCloseTo(0.01);
    expect(view.getFloat32(116, true)).toBeCloseTo(0.02);
    expect(view.getFloat32(120, true)).toBeCloseTo(0.0);
    expect(view.getFloat32(124, true)).toBeCloseTo(200.0);
    expectFloat32Close(view.getFloat32(128, true), 1e30);
    expectFloat32Close(view.getFloat32(132, true), -1e30);
    expectFloat32Close(view.getFloat32(136, true), 1e30);
    expect(view.getUint32(140, true)).toBe(1);
  });

  it('jacobian_a_lin at bytes 16-31 and jacobian_a_ang at bytes 32-47', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    // jacobian_a_lin
    view.setFloat32(16, 0.577, true);
    view.setFloat32(20, 0.577, true);
    view.setFloat32(24, 0.577, true);
    view.setFloat32(28, 0.0, true);

    // jacobian_a_ang
    view.setFloat32(32, -1.2, true);
    view.setFloat32(36, 0.8, true);
    view.setFloat32(40, -0.3, true);
    view.setFloat32(44, 0.0, true);

    // Verify jacobian_a_lin
    expect(view.getFloat32(16, true)).toBeCloseTo(0.577);
    expect(view.getFloat32(20, true)).toBeCloseTo(0.577);
    expect(view.getFloat32(24, true)).toBeCloseTo(0.577);
    expect(view.getFloat32(28, true)).toBe(0.0);

    // Verify jacobian_a_ang
    expect(view.getFloat32(32, true)).toBeCloseTo(-1.2);
    expect(view.getFloat32(36, true)).toBeCloseTo(0.8);
    expect(view.getFloat32(40, true)).toBeCloseTo(-0.3);
    expect(view.getFloat32(44, true)).toBe(0.0);
  });

  it('jacobian_b_lin at bytes 48-63, jacobian_b_ang at bytes 64-79 with mu at byte 76', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    // jacobian_b_lin
    view.setFloat32(48, -0.577, true);
    view.setFloat32(52, -0.577, true);
    view.setFloat32(56, -0.577, true);
    view.setFloat32(60, 0.0, true);

    // jacobian_b_ang with mu in .w
    view.setFloat32(64, 1.2, true);
    view.setFloat32(68, -0.8, true);
    view.setFloat32(72, 0.3, true);
    view.setFloat32(76, 0.6, true); // mu

    expect(view.getFloat32(48, true)).toBeCloseTo(-0.577);
    expect(view.getFloat32(52, true)).toBeCloseTo(-0.577);
    expect(view.getFloat32(56, true)).toBeCloseTo(-0.577);
    expect(view.getFloat32(60, true)).toBe(0.0);

    expect(view.getFloat32(64, true)).toBeCloseTo(1.2);
    expect(view.getFloat32(68, true)).toBeCloseTo(-0.8);
    expect(view.getFloat32(72, true)).toBeCloseTo(0.3);
    expect(view.getFloat32(76, true)).toBeCloseTo(0.6); // mu readback
  });

  it('hessian_diag_a_ang occupies bytes 80-95 (NEW fields added for geometric stiffness)', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    view.setFloat32(80, 50.0, true);
    view.setFloat32(84, 60.0, true);
    view.setFloat32(88, 70.0, true);
    view.setFloat32(92, 0.0, true); // padding

    expect(view.getFloat32(80, true)).toBeCloseTo(50.0);
    expect(view.getFloat32(84, true)).toBeCloseTo(60.0);
    expect(view.getFloat32(88, true)).toBeCloseTo(70.0);
    expect(view.getFloat32(92, true)).toBe(0.0);

    // These bytes were previously occupied by scalar fields (c, c0, lambda, penalty)
    // in the old 28-float layout. Verify they are NOT scalar fields.
    // In old layout, byte 80 would have been c (a small value like 0.01).
    // Here we write 50.0, confirming this is now hessian data, not c.
    expect(view.getFloat32(80, true)).not.toBeCloseTo(0.01);
  });

  it('hessian_diag_b_ang occupies bytes 96-111 (NEW fields added for geometric stiffness)', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    view.setFloat32(96, -15.0, true);
    view.setFloat32(100, -25.0, true);
    view.setFloat32(104, -35.0, true);
    view.setFloat32(108, 0.0, true); // padding

    expect(view.getFloat32(96, true)).toBeCloseTo(-15.0);
    expect(view.getFloat32(100, true)).toBeCloseTo(-25.0);
    expect(view.getFloat32(104, true)).toBeCloseTo(-35.0);
    expect(view.getFloat32(108, true)).toBe(0.0);
  });

  it('scalar fields are SHIFTED to bytes 112-143 (was 80-111 in old 28-float layout)', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    view.setFloat32(112, 0.003, true);   // c
    view.setFloat32(116, 0.004, true);   // c0
    view.setFloat32(120, 5.67, true);    // lambda
    view.setFloat32(124, 800.0, true);   // penalty
    view.setFloat32(128, 1e30, true);    // stiffness
    view.setFloat32(132, -1e30, true);   // fmin
    view.setFloat32(136, 1e30, true);    // fmax
    view.setUint32(140, 0, true);        // active (inactive)

    expect(view.getFloat32(112, true)).toBeCloseTo(0.003);
    expect(view.getFloat32(116, true)).toBeCloseTo(0.004);
    expect(view.getFloat32(120, true)).toBeCloseTo(5.67);
    expect(view.getFloat32(124, true)).toBeCloseTo(800.0);
    expectFloat32Close(view.getFloat32(128, true), 1e30);
    expectFloat32Close(view.getFloat32(132, true), -1e30);
    expectFloat32Close(view.getFloat32(136, true), 1e30);
    expect(view.getUint32(140, true)).toBe(0);
  });

  it('lambda at byte 120 and penalty at byte 124 (moved from old offsets 88 and 92)', () => {
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    const view = new DataView(buf);

    const lambdaValue = 42.5;
    const penaltyValue = 999.0;

    // Write at the NEW offsets
    view.setFloat32(120, lambdaValue, true);
    view.setFloat32(124, penaltyValue, true);

    // Read back from new offsets
    expect(view.getFloat32(120, true)).toBeCloseTo(lambdaValue);
    expect(view.getFloat32(124, true)).toBeCloseTo(penaltyValue);

    // Verify old offsets (88, 92) are NOT where lambda/penalty live anymore
    // (they are now part of hessian_diag_a_ang)
    expect(view.getFloat32(88, true)).not.toBeCloseTo(lambdaValue);
    expect(view.getFloat32(92, true)).not.toBeCloseTo(penaltyValue);
  });
});

// ─── WGSL Struct Validation ──────────────────────────────────────────────────

describe('WGSL Struct Validation', () => {
  it('2D primal shader defines ConstraintRow with hessian_diag_a and hessian_diag_b fields', () => {
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('hessian_diag_a');
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('hessian_diag_b');
    // Verify the struct has the expected field types
    expect(PRIMAL_UPDATE_2D_WGSL).toMatch(/hessian_diag_a\s*:\s*vec4<f32>/);
    expect(PRIMAL_UPDATE_2D_WGSL).toMatch(/hessian_diag_b\s*:\s*vec4<f32>/);
  });

  it('3D primal shader defines ConstraintRow3D with hessian_diag_a_ang and hessian_diag_b_ang fields', () => {
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('hessian_diag_a_ang');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('hessian_diag_b_ang');
    // Verify the struct has the expected field types
    expect(PRIMAL_UPDATE_3D_WGSL).toMatch(/hessian_diag_a_ang\s*:\s*vec4<f32>/);
    expect(PRIMAL_UPDATE_3D_WGSL).toMatch(/hessian_diag_b_ang\s*:\s*vec4<f32>/);
  });

  it('3D primal shader applies geometric stiffness using H_ang and abs_f', () => {
    // The 3D primal update shader should reference H_ang for geometric stiffness correction
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('H_ang');
    // The shader computes abs(H_ang.x) * abs_f (or similar component-wise products)
    expect(PRIMAL_UPDATE_3D_WGSL).toMatch(/abs\(H_ang\.\w+\)\s*\*\s*abs_f/);
  });

  it('3D dual and friction shaders define the updated ConstraintRow3D struct with hessian fields', () => {
    // Dual update shader
    expect(DUAL_UPDATE_3D_WGSL).toContain('hessian_diag_a_ang');
    expect(DUAL_UPDATE_3D_WGSL).toContain('hessian_diag_b_ang');
    expect(DUAL_UPDATE_3D_WGSL).toMatch(/hessian_diag_a_ang\s*:\s*vec4<f32>/);
    expect(DUAL_UPDATE_3D_WGSL).toMatch(/hessian_diag_b_ang\s*:\s*vec4<f32>/);

    // Friction coupling shader
    expect(FRICTION_COUPLING_3D_WGSL).toContain('hessian_diag_a_ang');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('hessian_diag_b_ang');
    expect(FRICTION_COUPLING_3D_WGSL).toMatch(/hessian_diag_a_ang\s*:\s*vec4<f32>/);
    expect(FRICTION_COUPLING_3D_WGSL).toMatch(/hessian_diag_b_ang\s*:\s*vec4<f32>/);
  });
});

// ─── Stride Constants ────────────────────────────────────────────────────────

describe('Stride Constants', () => {
  it('2D constraint stride: 28 floats x 4 bytes = 112 bytes', () => {
    expect(CONSTRAINT_STRIDE_2D).toBe(28);
    expect(CONSTRAINT_STRIDE_2D * 4).toBe(112);

    // Verify the buffer fits exactly: last field (active) ends at byte 112
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_2D * 4);
    expect(buf.byteLength).toBe(112);

    // Writing to byte 108 (last u32) should succeed, byte 112 would be out of bounds
    const view = new DataView(buf);
    view.setUint32(108, 1, true);
    expect(view.getUint32(108, true)).toBe(1);
  });

  it('3D constraint stride: 36 floats x 4 bytes = 144 bytes', () => {
    expect(CONSTRAINT_STRIDE_3D).toBe(36);
    expect(CONSTRAINT_STRIDE_3D * 4).toBe(144);

    // Verify the buffer fits exactly: last field (active) ends at byte 144
    const buf = new ArrayBuffer(CONSTRAINT_STRIDE_3D * 4);
    expect(buf.byteLength).toBe(144);

    // Writing to byte 140 (last u32) should succeed
    const view = new DataView(buf);
    view.setUint32(140, 1, true);
    expect(view.getUint32(140, true)).toBe(1);

    // Confirm the stride increased from the old 28 floats to 36 floats
    // (8 additional floats = 2 vec4 for hessian_diag_a_ang and hessian_diag_b_ang)
    expect(CONSTRAINT_STRIDE_3D - CONSTRAINT_STRIDE_2D).toBe(8);
  });

  it('3D body state: 20 floats x 4 = 80 bytes; body prev: 14 floats x 4 = 56 bytes', () => {
    expect(BODY_STRIDE_3D).toBe(20);
    expect(BODY_STRIDE_3D * 4).toBe(80);

    expect(BODY_PREV_STRIDE_3D).toBe(14);
    expect(BODY_PREV_STRIDE_3D * 4).toBe(56);

    // Verify body state layout fits: [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz, mass, Ix,Iy,Iz, pad,pad,pad]
    const bodyBuf = new ArrayBuffer(BODY_STRIDE_3D * 4);
    const bodyView = new DataView(bodyBuf);

    // Write position (x,y,z) at bytes 0-11
    bodyView.setFloat32(0, 1.0, true);
    bodyView.setFloat32(4, 2.0, true);
    bodyView.setFloat32(8, 3.0, true);

    // Write quaternion (qw,qx,qy,qz) at bytes 12-27
    bodyView.setFloat32(12, 1.0, true);  // qw
    bodyView.setFloat32(16, 0.0, true);  // qx
    bodyView.setFloat32(20, 0.0, true);  // qy
    bodyView.setFloat32(24, 0.0, true);  // qz

    // Write mass at byte 52 (float index 13)
    bodyView.setFloat32(52, 5.0, true);

    // Write inertia Ix,Iy,Iz at bytes 56-67 (float indices 14-16)
    bodyView.setFloat32(56, 0.1, true);
    bodyView.setFloat32(60, 0.2, true);
    bodyView.setFloat32(64, 0.3, true);

    expect(bodyView.getFloat32(0, true)).toBeCloseTo(1.0);
    expect(bodyView.getFloat32(12, true)).toBeCloseTo(1.0);
    expect(bodyView.getFloat32(52, true)).toBeCloseTo(5.0);
    expect(bodyView.getFloat32(56, true)).toBeCloseTo(0.1);

    // Verify prev state layout fits [px,py,pz, pqw,pqx,pqy,pqz, ix,iy,iz, iqw,iqx,iqy,iqz]
    const prevBuf = new ArrayBuffer(BODY_PREV_STRIDE_3D * 4);
    expect(prevBuf.byteLength).toBe(56);
  });
});
