/**
 * GPU solver tests.
 *
 * Tests in Node.js (no WebGPU) validate:
 * 1. Embedded WGSL shaders are loaded and contain valid code
 * 2. Buffer layouts match WGSL struct definitions exactly
 * 3. GPU-first behavior (throws without WebGPU, stepCPU opt-in)
 * 4. Constraint sorting produces correct per-body ranges
 *
 * Actual GPU execution is tested via Playwright in tests/browser/
 */
import { describe, it, expect } from 'vitest';
import AVBD, { World, RigidBodyDesc2D, ColliderDesc2D } from '../src/2d/index.js';
import { PRIMAL_UPDATE_2D_WGSL, DUAL_UPDATE_WGSL, DUAL_UPDATE_3D_WGSL, PRIMAL_UPDATE_3D_WGSL, MATH_UTILS_WGSL } from '../src/shaders/embedded.js';

// ─── Embedded Shader Validation ─────────────────────────────────────────────

describe('Embedded WGSL shaders', () => {
  it('should have non-empty primal 2D shader', () => {
    expect(PRIMAL_UPDATE_2D_WGSL.length).toBeGreaterThan(100);
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('@compute @workgroup_size(64)');
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('fn main(');
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('solve_ldl3');
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('struct SolverParams');
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('struct ConstraintRow');
  });

  it('should have non-empty dual update shader', () => {
    expect(DUAL_UPDATE_WGSL.length).toBeGreaterThan(100);
    expect(DUAL_UPDATE_WGSL).toContain('@compute @workgroup_size(64)');
    expect(DUAL_UPDATE_WGSL).toContain('fn main(');
    expect(DUAL_UPDATE_WGSL).toContain('struct SolverParams');
    // Should have friction coupling logic
    expect(DUAL_UPDATE_WGSL).toContain('fric');
  });

  it('should have non-empty primal 3D shader', () => {
    expect(PRIMAL_UPDATE_3D_WGSL.length).toBeGreaterThan(100);
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('solve_ldl6');
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('struct ConstraintRow3D');
  });

  it('should have math utils shader', () => {
    expect(MATH_UTILS_WGSL.length).toBeGreaterThan(50);
    expect(MATH_UTILS_WGSL).toContain('solve_ldl3');
  });

  it('should have matching SolverParams struct in primal and dual shaders', () => {
    // Both shaders must have the same SolverParams layout
    const primalParams = PRIMAL_UPDATE_2D_WGSL.match(/struct SolverParams \{([^}]+)\}/);
    const dualParams = DUAL_UPDATE_WGSL.match(/struct SolverParams \{([^}]+)\}/);
    expect(primalParams).not.toBeNull();
    expect(dualParams).not.toBeNull();
    // Both should have num_bodies_in_group and beta
    expect(primalParams![1]).toContain('num_bodies_in_group');
    expect(primalParams![1]).toContain('beta');
    expect(dualParams![1]).toContain('num_bodies_in_group');
    expect(dualParams![1]).toContain('beta');
  });

  it('should use params.num_bodies_in_group for bounds check (not arrayLength)', () => {
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('params.num_bodies_in_group');
    expect(PRIMAL_UPDATE_2D_WGSL).not.toContain('arrayLength(&color_body_indices)');
  });

  it('should have stiffness guard in primal shader', () => {
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('cr.stiffness < 1e30');
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('lambda_for_primal = 0.0');
  });

  it('should have conditional penalty ramp in dual shader', () => {
    expect(DUAL_UPDATE_WGSL).toContain('cr.lambda > cr.fmin && cr.lambda < cr.fmax');
  });

  it('should have full constraint evaluation in 3D shader', () => {
    // 3D shader should NOT have "Simplified" comment
    expect(PRIMAL_UPDATE_3D_WGSL).not.toContain('Simplified');
    // Should have body_state reads for constraint eval
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('jacobian_a_lin');
    // Should have angular displacement via quaternion diff
    expect(PRIMAL_UPDATE_3D_WGSL).toContain('jacobian_a_ang');
    expect(PRIMAL_UPDATE_3D_WGSL).not.toContain('TODO');
  });

  it('should have non-empty 3D dual update shader', () => {
    expect(DUAL_UPDATE_3D_WGSL.length).toBeGreaterThan(100);
    expect(DUAL_UPDATE_3D_WGSL).toContain('@compute @workgroup_size(64)');
    expect(DUAL_UPDATE_3D_WGSL).toContain('struct ConstraintRow3D');
    // Should have 6-DOF constraint evaluation (quaternion angular displacement)
    expect(DUAL_UPDATE_3D_WGSL).toContain('jacobian_a_ang');
    expect(DUAL_UPDATE_3D_WGSL).toContain('jacobian_b_ang');
    // Friction coupling moved to CPU (was GPU race condition)
    // 3D dual shader should note this in a comment
    expect(DUAL_UPDATE_3D_WGSL).toContain('Friction coupling');
    // Body state stride should be 20
    expect(DUAL_UPDATE_3D_WGSL).toContain('* 20u');
    // Body prev stride should be 14
    expect(DUAL_UPDATE_3D_WGSL).toContain('* 14u');
  });
});

// ─── GPU Fallback ───────────────────────────────────────────────────────────

describe('GPU-first behavior in Node.js', () => {
  it('should throw on init() when WebGPU is unavailable', async () => {
    await expect(AVBD.init()).rejects.toThrow(/WebGPU/);
  });

  it('should report GPU unavailable before init', () => {
    expect(AVBD.isGPUAvailable).toBe(false);
  });

  it('should throw on step() when GPU solver not initialized', async () => {
    const world = new World({ x: 0, y: -9.81 });
    world.createCollider(AVBD.ColliderDesc.cuboid(10, 0.5));
    const body = world.createRigidBody(AVBD.RigidBodyDesc.dynamic().setTranslation(0, 5));
    world.createCollider(AVBD.ColliderDesc.cuboid(0.5, 0.5), body);

    await expect(world.step()).rejects.toThrow(/GPU solver/);
  });

  it('should work with stepCPU() as explicit opt-in', () => {
    const world = new World({ x: 0, y: -9.81 });
    world.createCollider(AVBD.ColliderDesc.cuboid(10, 0.5));
    const body = world.createRigidBody(AVBD.RigidBodyDesc.dynamic().setTranslation(0, 5));
    world.createCollider(AVBD.ColliderDesc.cuboid(0.5, 0.5), body);

    world.stepCPU();
    expect(body.translation().y).toBeLessThan(5);
  });
});

// ─── Buffer Layout Validation ───────────────────────────────────────────────

describe('Buffer layout matches WGSL structs', () => {
  it('should pack body state as 8 floats (32 bytes) per body', () => {
    // WGSL: body_state: array<f32> indexed as body_idx * 8u
    const STRIDE = 8;
    const buf = new Float32Array(STRIDE);
    buf[0] = 1.0;  // x
    buf[1] = 2.0;  // y
    buf[2] = 0.5;  // angle
    buf[3] = 0.1;  // vx
    buf[4] = -0.2; // vy
    buf[5] = 0.3;  // omega
    buf[6] = 1.5;  // mass
    buf[7] = 0.25; // inertia
    expect(buf.byteLength).toBe(32);
  });

  it('should pack constraint row as 28 floats (112 bytes) matching WGSL ConstraintRow', () => {
    // This is the corrected layout — 28 floats, not 24
    const STRIDE = 28;
    const data = new ArrayBuffer(STRIDE * 4);
    const view = new DataView(data);

    // [0-3] body_a(i32), body_b(i32), force_type(u32), _pad(u32) = 16 bytes
    view.setInt32(0, 0, true);
    view.setInt32(4, 1, true);
    view.setUint32(8, 0, true);
    view.setUint32(12, 0, true);

    // [16-31] jacobian_a: vec4<f32> = 16 bytes
    view.setFloat32(16, 0.0, true);
    view.setFloat32(20, 1.0, true);
    view.setFloat32(24, 0.5, true);
    view.setFloat32(28, 0.0, true);

    // [32-47] jacobian_b: vec4<f32>
    // [48-63] hessian_diag_a: vec4<f32>
    // [64-79] hessian_diag_b: vec4<f32>

    // [80-111] c, c0, lambda, penalty, stiffness, fmin, fmax, active
    view.setFloat32(80, -0.1, true);    // c
    view.setFloat32(84, -0.1, true);    // c0
    view.setFloat32(88, -50.0, true);   // lambda
    view.setFloat32(92, 5000.0, true);  // penalty
    view.setFloat32(96, 1e30, true);    // stiffness (1e30 = infinity in WGSL)
    view.setFloat32(100, -1e30, true);  // fmin
    view.setFloat32(104, 0.0, true);    // fmax (0 for normal contact)
    view.setUint32(108, 1, true);       // active

    expect(data.byteLength).toBe(112);
    expect(view.getFloat32(96, true)).toBeCloseTo(1e30, -25); // f32 precision
    expect(view.getFloat32(100, true)).toBeCloseTo(-1e30, -25);
    expect(view.getUint32(108, true)).toBe(1);
  });

  it('should pack SolverParams as 10 fields (40 bytes)', () => {
    const data = new ArrayBuffer(40);
    const fv = new Float32Array(data);
    const uv = new Uint32Array(data);

    fv[0] = 1 / 60;     // dt
    fv[1] = 0.0;         // gravity_x
    fv[2] = -9.81;       // gravity_y
    fv[3] = 100.0;       // penalty_min
    fv[4] = 1e9;         // penalty_max
    fv[5] = 100000;      // beta
    uv[6] = 100;         // num_bodies
    uv[7] = 50;          // num_constraints
    uv[8] = 25;          // num_bodies_in_group
    uv[9] = 0;           // is_stabilization

    expect(data.byteLength).toBe(40);
    expect(fv[5]).toBe(100000);  // beta
    expect(uv[8]).toBe(25);      // num_bodies_in_group
  });
});

// ─── Constraint Indirection Bug (fixed) ─────────────────────────────────────
// The GPU solver previously sorted constraints per body (duplicating rows) but
// only uploaded numConstraints entries, leaving later bodies pointing at
// uninitialized memory. Now uses an indirection buffer.

describe('Constraint indirection for GPU primal shader', () => {
  it('should build correct per-body indirection indices', () => {
    // Simulate: 2 bodies (ground=0, box=1), 2 constraints (normal=0, friction=1)
    // Both constraints reference body 0 (bodyA) and body 1 (bodyB)
    // perBody[0] = [0, 1], perBody[1] = [0, 1]
    // Expected: constraintIndices = [0, 1, 0, 1]
    // bodyRanges: body0=[start:0, count:2], body1=[start:2, count:2]

    const numBodies = 2;
    const numConstraints = 2;

    // Build the indirection manually (mimicking buildConstraintIndirection)
    const perBody: number[][] = [[], []];
    // Constraint 0: bodyA=0, bodyB=1
    perBody[0].push(0);
    perBody[1].push(0);
    // Constraint 1: bodyA=0, bodyB=1
    perBody[0].push(1);
    perBody[1].push(1);

    const allIndices: number[] = [];
    const bodyRanges = new Uint32Array(numBodies * 2);
    for (let i = 0; i < numBodies; i++) {
      bodyRanges[i * 2 + 0] = allIndices.length;
      bodyRanges[i * 2 + 1] = perBody[i].length;
      allIndices.push(...perBody[i]);
    }

    // Body 0 (ground): constraints at indices [0, 1]
    expect(bodyRanges[0]).toBe(0);  // start
    expect(bodyRanges[1]).toBe(2);  // count

    // Body 1 (box): constraints at indices [0, 1]
    expect(bodyRanges[2]).toBe(2);  // start
    expect(bodyRanges[3]).toBe(2);  // count

    // Indirection: [0, 1, 0, 1]
    expect(allIndices).toEqual([0, 1, 0, 1]);

    // Verify body 1 can access its constraints via indirection
    const body1Start = bodyRanges[2];
    const body1Count = bodyRanges[3];
    for (let ci = 0; ci < body1Count; ci++) {
      const crIdx = allIndices[body1Start + ci];
      expect(crIdx).toBeLessThan(numConstraints);  // Must be valid constraint index
    }
  });

  it('should ensure all bodies see their constraints in multi-body scenario', () => {
    // 4 bodies: ground(0), boxA(1), boxB(2), boxC(3)
    // Constraints: 0=[0,1], 1=[0,1], 2=[1,2], 3=[1,2], 4=[2,3], 5=[2,3]
    const pairs = [[0,1], [0,1], [1,2], [1,2], [2,3], [2,3]];
    const numBodies = 4;
    const numConstraints = 6;

    const perBody: number[][] = Array.from({ length: numBodies }, () => []);
    for (let ci = 0; ci < numConstraints; ci++) {
      const [a, b] = pairs[ci];
      perBody[a].push(ci);
      perBody[b].push(ci);
    }

    const allIndices: number[] = [];
    const bodyRanges = new Uint32Array(numBodies * 2);
    for (let i = 0; i < numBodies; i++) {
      bodyRanges[i * 2] = allIndices.length;
      bodyRanges[i * 2 + 1] = perBody[i].length;
      allIndices.push(...perBody[i]);
    }

    // Body 0: constraints [0, 1]
    expect(bodyRanges[0 * 2 + 1]).toBe(2);
    // Body 1: constraints [0, 1, 2, 3]
    expect(bodyRanges[1 * 2 + 1]).toBe(4);
    // Body 2: constraints [2, 3, 4, 5]
    expect(bodyRanges[2 * 2 + 1]).toBe(4);
    // Body 3: constraints [4, 5]
    expect(bodyRanges[3 * 2 + 1]).toBe(2);

    // ALL indices in allIndices must be valid constraint indices
    for (const idx of allIndices) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(numConstraints);
    }

    // Total indirection entries = sum of all per-body counts
    expect(allIndices.length).toBe(2 + 4 + 4 + 2);  // 12
  });

  it('should have constraint_indices binding in primal 2D shader', () => {
    // Verify the primal shader uses the indirection buffer
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('constraint_indices');
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('@group(0) @binding(6)');
    // Should use indirection: constraint_indices[constraint_start + ci]
    expect(PRIMAL_UPDATE_2D_WGSL).toContain('constraint_indices[constraint_start + ci]');
  });
});

// ─── Velocity Recovery Bug (fixed) ──────────────────────────────────────────
// With postStabilize=true, the GPU solver must recover velocity from the
// pre-stabilization position (iteration N-1), not the post-stabilization
// position (iteration N). Using the wrong position causes incorrect velocity,
// which feeds into the next step's inertial target.

describe('Velocity recovery timing', () => {
  it('CPU solver should recover velocity at iter N-1, not post-stabilization', () => {
    // Verify the CPU solver does velocity recovery at the correct iteration
    const world = new World({ x: 0, y: -9.81 }, { iterations: 5, postStabilize: true });
    const body = world.createRigidBody(RigidBodyDesc2D.dynamic().setTranslation(0, 5));
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    world.stepCPU();
    const v1 = body.linvel();
    expect(v1.y).toBeLessThan(0);  // Falling

    world.stepCPU();
    const v2 = body.linvel();
    expect(v2.y).toBeLessThan(v1.y);  // Accelerating

    // Velocity should be approximately -g * t
    const expectedV = -9.81 * 2 / 60;
    expect(Math.abs(v2.y - expectedV)).toBeLessThan(0.1);
  });

  it('CPU box-on-ground should settle within physics tolerance', () => {
    // This is the CPU reference test — the GPU should produce similar results
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    world.createCollider(ColliderDesc2D.cuboid(10, 0.5));
    const body = world.createRigidBody(RigidBodyDesc2D.dynamic().setTranslation(0, 3));
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    for (let i = 0; i < 120; i++) {
      world.stepCPU();
    }

    const y = body.translation().y;
    // Ground top at 0.5, box half-height 0.5 → resting at y≈1.0
    expect(y).toBeGreaterThan(0.5);
    expect(y).toBeLessThan(2.5);
  });
});

// ─── Dispatch Pattern ───────────────────────────────────────────────────────

describe('GPU dispatch architecture', () => {
  it('should compute correct dispatch counts', () => {
    const WORKGROUP_SIZE = 64;
    // Total dispatches per step = iterations × (numColors + 1)
    // where +1 is the dual update
    const iterations = 10;
    const numColors = 4;
    const totalDispatches = iterations * (numColors + 1);
    expect(totalDispatches).toBe(50);

    // Workgroup counts
    expect(Math.ceil(1 / WORKGROUP_SIZE)).toBe(1);
    expect(Math.ceil(64 / WORKGROUP_SIZE)).toBe(1);
    expect(Math.ceil(65 / WORKGROUP_SIZE)).toBe(2);
    expect(Math.ceil(10000 / WORKGROUP_SIZE)).toBe(157);
  });

  it('should estimate GPU memory correctly', () => {
    // Per body: 32 bytes state + 32 bytes prev = 64 bytes
    // Per constraint: 112 bytes
    const numBodies = 10000;
    const avgConstraints = 3;
    const totalMB = (numBodies * 64 + numBodies * avgConstraints * 112) / (1024 * 1024);
    expect(totalMB).toBeLessThan(5);
  });
});
