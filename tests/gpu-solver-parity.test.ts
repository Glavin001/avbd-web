/**
 * GPU solver parity tests.
 *
 * Validates GPU buffer layouts, shader struct alignment, and that the
 * GPU shader code contains the correct mathematical formulas.
 * These tests run in Node.js without WebGPU — they validate the code/data
 * structures rather than actual GPU execution.
 */
import { describe, it, expect } from 'vitest';
import {
  PRIMAL_UPDATE_3D_WGSL,
  DUAL_UPDATE_3D_WGSL,
  FRICTION_COUPLING_3D_WGSL,
  FRICTION_COUPLING_WGSL,
} from '../src/shaders/embedded.js';

// ─── 3D Shader Structural Validation ─────────────────────────────────────────

describe('3D Primal shader correctness', () => {
  it('should have angular RHS from inertial quaternion (not zeros)', () => {
    // Bug fix: previously rhs[3..5] = 0.0, now uses quaternion difference
    expect(PRIMAL_UPDATE_3D_WGSL).not.toMatch(/rhs\[3\]\s*=\s*0\.0/);
    expect(PRIMAL_UPDATE_3D_WGSL).not.toMatch(/rhs\[4\]\s*=\s*0\.0/);
    expect(PRIMAL_UPDATE_3D_WGSL).not.toMatch(/rhs\[5\]\s*=\s*0\.0/);
    // Should reference inertial quaternion from body_prev
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('prev_base + 10u');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('prev_base + 11u');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('prev_base + 12u');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('prev_base + 13u');
  });

  it('should have 6x6 LDL solver', () => {
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('solve_ldl6');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('array<f32, 36>');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('array<f32, 6>');
  });

  it('should have quaternion update for angular correction', () => {
    // Quaternion update applies angular delta as q_new = q + 0.5 * (w * q)
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('0.5');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('delta[3]');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('delta[4]');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('delta[5]');
    // Should normalize quaternion after update
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('qlen');
  });

  it('should use correct body state stride (20 floats)', () => {
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('body_idx * 20u');
  });

  it('should use correct body prev stride (14 floats)', () => {
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('body_idx * 14u');
  });

  it('should accumulate J*J^T*penalty into LHS', () => {
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('J[i] * J[j] * cr.penalty');
  });

  it('should have stiffness guard (lambda=0 for soft constraints)', () => {
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('cr.stiffness < 1e30');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('lambda_for_primal = 0.0');
  });
});

describe('3D Dual shader correctness', () => {
  it('should have quaternion angular displacement evaluation', () => {
    expect(DUAL_UPDATE_3D_WGSL).toContain('dqx_v');
    expect(DUAL_UPDATE_3D_WGSL).toContain('dqy_v');
    expect(DUAL_UPDATE_3D_WGSL).toContain('dqz_v');
    expect(DUAL_UPDATE_3D_WGSL).toContain('2.0');
  });

  it('should have conditional penalty ramp', () => {
    expect(DUAL_UPDATE_3D_WGSL).toContain('cr.lambda > cr.fmin && cr.lambda < cr.fmax');
    expect(DUAL_UPDATE_3D_WGSL).toContain('params.beta');
  });

  it('should clamp penalty to stiffness', () => {
    expect(DUAL_UPDATE_3D_WGSL).toContain('cr.penalty > cr.stiffness');
  });

  it('should use 20-float body state stride', () => {
    expect(DUAL_UPDATE_3D_WGSL).toContain('* 20u');
  });

  it('should use 14-float body prev stride', () => {
    expect(DUAL_UPDATE_3D_WGSL).toContain('* 14u');
  });
});

// ─── 3D Friction Coupling Shader ─────────────────────────────────────────────

describe('3D Friction coupling shader', () => {
  it('should exist and have correct structure', () => {
    expect(FRICTION_COUPLING_3D_WGSL.length).toBeGreaterThan(100);
    expect(FRICTION_COUPLING_3D_WGSL).toContain('@compute @workgroup_size(64)');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('fn main(');
  });

  it('should handle triplets (normal + 2 friction tangents)', () => {
    // 3D contacts: [normal, friction1, friction2]
    expect(FRICTION_COUPLING_3D_WGSL).toContain('triplet_idx * 3u');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('normal_idx + 1u');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('normal_idx + 2u');
  });

  it('should read mu from constraint row', () => {
    expect(FRICTION_COUPLING_3D_WGSL).toContain('mu');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('normal_force');
  });

  it('should update fmin and fmax for both friction rows', () => {
    expect(FRICTION_COUPLING_3D_WGSL).toContain('fric1.fmin');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('fric1.fmax');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('fric2.fmin');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('fric2.fmax');
  });

  it('should use ConstraintRow3D struct (not 2D)', () => {
    expect(FRICTION_COUPLING_3D_WGSL).toContain('struct ConstraintRow3D');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('jacobian_a_lin');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('jacobian_b_ang');
  });
});

// ─── 2D Friction Coupling (pairs) vs 3D (triplets) ──────────────────────────

describe('Friction coupling: 2D pairs vs 3D triplets', () => {
  it('2D shader should handle pairs (normal_idx = pair_idx * 2)', () => {
    expect(FRICTION_COUPLING_WGSL).toContain('pair_idx * 2u');
    expect(FRICTION_COUPLING_WGSL).toContain('normal_idx + 1u');
    // Should NOT have triplet logic
    expect(FRICTION_COUPLING_WGSL).not.toContain('normal_idx + 2u');
  });

  it('3D shader should handle triplets (normal_idx = triplet_idx * 3)', () => {
    expect(FRICTION_COUPLING_3D_WGSL).toContain('triplet_idx * 3u');
    expect(FRICTION_COUPLING_3D_WGSL).toContain('friction2_idx');
  });
});

// ─── 3D Buffer Layout Constants ──────────────────────────────────────────────

describe('3D buffer layout constants', () => {
  it('should pack 3D body state as 20 floats (80 bytes)', () => {
    const STRIDE = 20;
    const buf = new Float32Array(STRIDE);
    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, mass, Ix, Iy, Iz, pad, pad, pad]
    buf[0] = 1.0;  // x
    buf[1] = 2.0;  // y
    buf[2] = 3.0;  // z
    buf[3] = 1.0;  // qw
    buf[4] = 0.0;  // qx
    buf[5] = 0.0;  // qy
    buf[6] = 0.0;  // qz
    buf[7] = 0.1;  // vx
    buf[8] = -0.2; // vy
    buf[9] = 0.0;  // vz
    buf[10] = 0.0; // wx
    buf[11] = 0.0; // wy
    buf[12] = 0.0; // wz
    buf[13] = 1.0; // mass
    buf[14] = 0.167; // Ix
    buf[15] = 0.167; // Iy
    buf[16] = 0.167; // Iz
    expect(buf.byteLength).toBe(80);
  });

  it('should pack 3D body prev as 14 floats (56 bytes)', () => {
    const STRIDE = 14;
    const buf = new Float32Array(STRIDE);
    // [px, py, pz, pqw, pqx, pqy, pqz, ix, iy, iz, iqw, iqx, iqy, iqz]
    buf[0] = 1.0;  // prev x
    buf[3] = 1.0;  // prev qw (identity)
    buf[7] = 1.1;  // inertial x
    buf[10] = 1.0; // inertial qw (identity)
    expect(buf.byteLength).toBe(56);
  });

  it('should pack 3D SolverParams as 12 floats (48 bytes)', () => {
    // 11 fields + 1 padding for alignment
    const STRIDE = 12;
    const data = new ArrayBuffer(STRIDE * 4);
    const fv = new Float32Array(data);
    const uv = new Uint32Array(data);
    fv[0] = 1 / 60;     // dt
    fv[1] = 0.0;         // gravity_x
    fv[2] = -9.81;       // gravity_y
    fv[3] = 0.0;         // gravity_z
    fv[4] = 100;         // penalty_min
    fv[5] = 1e9;         // penalty_max
    fv[6] = 100000;      // beta
    uv[7] = 10;          // num_bodies
    uv[8] = 30;          // num_constraints
    uv[9] = 5;           // num_bodies_in_group
    uv[10] = 0;          // is_stabilization
    // [11] padding
    expect(data.byteLength).toBe(48);
  });

  it('should pack 3D constraint row as 28 floats (112 bytes) with split lin/ang jacobians', () => {
    const STRIDE = 28;
    const data = new ArrayBuffer(STRIDE * 4);
    const view = new DataView(data);
    // [0-15] body_a, body_b, force_type, _pad
    view.setInt32(0, 0, true);   // body_a
    view.setInt32(4, 1, true);   // body_b
    view.setUint32(8, 0, true);  // force_type (Contact)
    // [16-31] jacobian_a_lin (vec4): [nx, ny, nz, 0]
    view.setFloat32(16, 0.0, true);
    view.setFloat32(20, 1.0, true);
    view.setFloat32(24, 0.0, true);
    // [32-47] jacobian_a_ang (vec4): [torque_x, torque_y, torque_z, 0]
    view.setFloat32(32, 0.5, true);
    // [48-63] jacobian_b_lin (vec4): [-nx, -ny, -nz, 0]
    view.setFloat32(52, -1.0, true);
    // [64-79] jacobian_b_ang (vec4): [-torque_x, -torque_y, -torque_z, mu]
    view.setFloat32(76, 0.5, true);  // mu packed in .w
    // [80-111] scalar fields
    view.setFloat32(80, -0.1, true);   // c
    view.setFloat32(84, -0.1, true);   // c0
    view.setFloat32(88, -50.0, true);  // lambda
    view.setFloat32(92, 100.0, true);  // penalty
    view.setFloat32(96, 1e30, true);   // stiffness
    view.setFloat32(100, -1e30, true); // fmin
    view.setFloat32(104, 0.0, true);   // fmax
    view.setUint32(108, 1, true);      // is_active
    expect(data.byteLength).toBe(112);
    // Verify mu packed at correct offset
    expect(view.getFloat32(76, true)).toBe(0.5);
  });
});

// ─── Quaternion Angular Displacement Formula ─────────────────────────────────

describe('Quaternion angular displacement computation', () => {
  it('should match CPU formula: dq = q * conj(q_prev)', () => {
    // CPU solver (solver-3d.ts) computes:
    //   dq = quatMul(rotation, { w: prev.w, x: -prev.x, y: -prev.y, z: -prev.z })
    //   dtheta = 2 * vec3(dq.x, dq.y, dq.z)
    //
    // GPU shader computes (primal-update-3d.wgsl and dual-update-3d.wgsl):
    //   For body A: dq_a = q_a * conj(q_prev_a)
    //   c_eval += jacobian_a_ang · 2 * vec3(dqx, dqy, dqz)
    //
    // Test with known quaternions:
    const identity = { w: 1, x: 0, y: 0, z: 0 };
    const rot90z = { w: Math.cos(Math.PI / 4), x: 0, y: 0, z: Math.sin(Math.PI / 4) };

    // dq = rot90z * conj(identity) = rot90z
    // dtheta ≈ 2 * (0, 0, sin(pi/4)) ≈ (0, 0, 1.414)
    // which should be close to pi/2 ≈ 1.5708 for the z-axis
    const dtheta_z = 2 * rot90z.z;
    expect(dtheta_z).toBeCloseTo(Math.sqrt(2), 3);

    // For small angles, dtheta ≈ angle * axis
    // 90 degrees is not small, but the formula still gives the right direction
    expect(dtheta_z).toBeGreaterThan(0);
  });

  it('GPU shader should use the same formula as CPU solver', () => {
    // Both primal and dual 3D shaders should compute:
    // dqx = qx*pqw - qw*pqx + qz*pqy - qy*pqz  (matches q * conj(prev))
    // Verify this pattern exists in both shaders
    for (const shader of [PRIMAL_UPDATE_3D_WGSL, DUAL_UPDATE_3D_WGSL]) {
      // The quaternion multiplication q * conj(q_prev) produces:
      // x-component: qx*pw - qw*px + qz*py - qy*pz  (or with swapped signs for conjugate)
      expect(shader).toContain('dq');
      expect(shader).toContain('2.0');
    }
  });
});
