// AUTO-GENERATED — do not edit. Run: npx tsx scripts/embed-shaders.ts

export const DUAL_UPDATE_3D_WGSL = `// ─── AVBD 3D Dual Update Compute Shader ────────────────────────────────────
// Updates lambda (Lagrange multiplier) and ramps penalty for each constraint.
// Each thread handles one constraint row.
// 3D variant: uses 20-float body state stride, 14-float prev stride,
// and 6-DOF constraint evaluation with quaternion angular displacement.
// Friction coupling: normal contact rows (index % 3 == 0) update adjacent
// two friction tangent rows.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  gravity_z: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
}

struct ConstraintRow3D {
  body_a: i32,
  body_b: i32,
  force_type: u32,
  _pad0: u32,
  jacobian_a_lin: vec4<f32>,
  jacobian_a_ang: vec4<f32>,
  jacobian_b_lin: vec4<f32>,
  jacobian_b_ang: vec4<f32>,
  c: f32,
  c0: f32,
  lambda: f32,
  penalty: f32,
  stiffness: f32,
  fmin: f32,
  fmax: f32,
  is_active: u32,
}

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> constraints: array<ConstraintRow3D>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_constraints) { return; }

  var cr = constraints[idx];
  if (cr.is_active == 0u) { return; }

  // Re-evaluate linearized constraint: C = C0 + J_A·dp_A + J_B·dp_B
  var c_eval = cr.c0;

  if (cr.body_a >= 0) {
    let ba = u32(cr.body_a) * 20u;
    let bap = u32(cr.body_a) * 14u;
    // Linear displacement
    let dpx = body_state[ba + 0u] - body_prev[bap + 0u];
    let dpy = body_state[ba + 1u] - body_prev[bap + 1u];
    let dpz = body_state[ba + 2u] - body_prev[bap + 2u];
    c_eval += cr.jacobian_a_lin.x * dpx + cr.jacobian_a_lin.y * dpy + cr.jacobian_a_lin.z * dpz;
    // Angular displacement via quaternion difference
    let qw = body_state[ba + 3u]; let qx = body_state[ba + 4u];
    let qy = body_state[ba + 5u]; let qz = body_state[ba + 6u];
    let pqw = body_prev[bap + 3u]; let pqx = body_prev[bap + 4u];
    let pqy = body_prev[bap + 5u]; let pqz = body_prev[bap + 6u];
    // dq = q * conj(q_prev): small angle -> dtheta ≈ 2 * vec3(dq.xyz)
    let dqw = pqw * qw + pqx * qx + pqy * qy + pqz * qz;
    let dqx_v = pqw * qx - pqx * qw - pqy * qz + pqz * qy;
    let dqy_v = pqw * qy + pqx * qz - pqy * qw - pqz * qx;
    let dqz_v = pqw * qz - pqx * qy + pqy * qx - pqz * qw;
    c_eval += cr.jacobian_a_ang.x * 2.0 * dqx_v
           + cr.jacobian_a_ang.y * 2.0 * dqy_v
           + cr.jacobian_a_ang.z * 2.0 * dqz_v;
  }

  if (cr.body_b >= 0) {
    let bb = u32(cr.body_b) * 20u;
    let bbp = u32(cr.body_b) * 14u;
    let dpx = body_state[bb + 0u] - body_prev[bbp + 0u];
    let dpy = body_state[bb + 1u] - body_prev[bbp + 1u];
    let dpz = body_state[bb + 2u] - body_prev[bbp + 2u];
    c_eval += cr.jacobian_b_lin.x * dpx + cr.jacobian_b_lin.y * dpy + cr.jacobian_b_lin.z * dpz;
    let qw = body_state[bb + 3u]; let qx = body_state[bb + 4u];
    let qy = body_state[bb + 5u]; let qz = body_state[bb + 6u];
    let pqw = body_prev[bbp + 3u]; let pqx = body_prev[bbp + 4u];
    let pqy = body_prev[bbp + 5u]; let pqz = body_prev[bbp + 6u];
    let dqw = pqw * qw + pqx * qx + pqy * qy + pqz * qz;
    let dqx_v = pqw * qx - pqx * qw - pqy * qz + pqz * qy;
    let dqy_v = pqw * qy + pqx * qz - pqy * qw - pqz * qx;
    let dqz_v = pqw * qz - pqx * qy + pqy * qx - pqz * qw;
    c_eval += cr.jacobian_b_ang.x * 2.0 * dqx_v
           + cr.jacobian_b_ang.y * 2.0 * dqy_v
           + cr.jacobian_b_ang.z * 2.0 * dqz_v;
  }

  // Stiffness guard: soft constraints zero lambda before accumulation
  var prev_lambda = cr.lambda;
  if (cr.stiffness < 1e30) {
    prev_lambda = 0.0;
  }

  // Update lambda
  var new_lambda = cr.penalty * c_eval + prev_lambda;
  new_lambda = clamp(new_lambda, cr.fmin, cr.fmax);
  cr.lambda = new_lambda;

  // Conditional penalty ramp: only when constraint is interior
  if (cr.lambda > cr.fmin && cr.lambda < cr.fmax) {
    cr.penalty += params.beta * abs(c_eval);
  }
  cr.penalty = clamp(cr.penalty, params.penalty_min, params.penalty_max);
  if (cr.penalty > cr.stiffness) {
    cr.penalty = cr.stiffness;
  }

  // Write back this row
  constraints[idx] = cr;

  // NOTE: Friction coupling (normal→friction bounds update) is computed on
  // CPU before upload. GPU parallel dispatch creates a race condition where
  // the friction thread reads stale bounds before the normal thread writes.
}
`;

export const DUAL_UPDATE_WGSL = `// ─── AVBD Dual Update Compute Shader ────────────────────────────────────────
// Updates lambda (Lagrange multiplier) and ramps penalty for each constraint.
// Each thread handles one constraint row.
// Includes friction coupling: normal contact rows update adjacent friction rows.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
}

struct ConstraintRow {
  body_a: i32,
  body_b: i32,
  force_type: u32,
  _pad0: u32,
  jacobian_a: vec4<f32>,
  jacobian_b: vec4<f32>,
  hessian_diag_a: vec4<f32>,
  hessian_diag_b: vec4<f32>,
  c: f32,
  c0: f32,
  lambda: f32,
  penalty: f32,
  stiffness: f32,
  fmin: f32,
  fmax: f32,
  is_active: u32,
}

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> constraints: array<ConstraintRow>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_constraints) { return; }

  var cr = constraints[idx];
  if (cr.is_active == 0u) { return; }

  // Re-evaluate linearized constraint
  var c_eval = cr.c0;
  if (cr.body_a >= 0) {
    let ba_base = u32(cr.body_a) * 8u;
    c_eval += cr.jacobian_a.x * (body_state[ba_base + 0u] - body_prev[ba_base + 0u])
           + cr.jacobian_a.y * (body_state[ba_base + 1u] - body_prev[ba_base + 1u])
           + cr.jacobian_a.z * (body_state[ba_base + 2u] - body_prev[ba_base + 2u]);
  }
  if (cr.body_b >= 0) {
    let bb_base = u32(cr.body_b) * 8u;
    c_eval += cr.jacobian_b.x * (body_state[bb_base + 0u] - body_prev[bb_base + 0u])
           + cr.jacobian_b.y * (body_state[bb_base + 1u] - body_prev[bb_base + 1u])
           + cr.jacobian_b.z * (body_state[bb_base + 2u] - body_prev[bb_base + 2u]);
  }

  // Stiffness guard: soft constraints zero lambda before accumulation
  var prev_lambda = cr.lambda;
  if (cr.stiffness < 1e30) {
    prev_lambda = 0.0;
  }

  // Update lambda
  var new_lambda = cr.penalty * c_eval + prev_lambda;
  new_lambda = clamp(new_lambda, cr.fmin, cr.fmax);
  cr.lambda = new_lambda;

  // Conditional penalty ramp: only when constraint is interior
  if (cr.lambda > cr.fmin && cr.lambda < cr.fmax) {
    cr.penalty += params.beta * abs(c_eval);
  }
  cr.penalty = clamp(cr.penalty, params.penalty_min, params.penalty_max);
  if (cr.penalty > cr.stiffness) {
    cr.penalty = cr.stiffness;
  }

  // Write back this row
  constraints[idx] = cr;

  // NOTE: Friction coupling runs as a SEPARATE compute pass after the dual
  // update, guaranteeing all normal lambdas are finalized before friction
  // bounds are updated. See FRICTION_COUPLING_WGSL.
}
`;

// ─── Friction Coupling Shader ───────────────────────────────────────────────
// Runs as a separate compute pass AFTER the dual update.
// Per the AVBD reference (manifold.cpp: computeConstraint), friction bounds
// are updated using the current normal lambda BEFORE the next iteration's
// primal/dual. Running this after dual guarantees all lambdas are finalized.
// Contact rows are ordered [normal, friction, normal, friction, ...].
// One thread per contact PAIR (not per row).
export const FRICTION_COUPLING_WGSL = `
struct FrictionParams {
  num_constraints: u32,
}

struct ConstraintRow {
  body_a: i32,
  body_b: i32,
  force_type: u32,
  _pad0: u32,
  jacobian_a: vec4<f32>,
  jacobian_b: vec4<f32>,
  hessian_diag_a: vec4<f32>,
  hessian_diag_b: vec4<f32>,
  c: f32,
  c0: f32,
  lambda: f32,
  penalty: f32,
  stiffness: f32,
  fmin: f32,
  fmax: f32,
  is_active: u32,
}

@group(0) @binding(0) var<uniform> params: FrictionParams;
@group(0) @binding(1) var<storage, read_write> constraints: array<ConstraintRow>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_idx = gid.x;
  let normal_idx = pair_idx * 2u;
  let friction_idx = normal_idx + 1u;

  if (friction_idx >= params.num_constraints) { return; }

  let normal = constraints[normal_idx];
  // Only process contact pairs (force_type 0 = Contact)
  if (normal.force_type != 0u || normal.is_active == 0u) { return; }

  var friction = constraints[friction_idx];
  if (friction.force_type != 0u || friction.is_active == 0u) { return; }

  // mu is packed in hessian_diag_b.w of the normal row during CPU upload
  let mu = normal.hessian_diag_b.w;
  let normal_force = abs(normal.lambda);
  friction.fmin = -mu * normal_force;
  friction.fmax = mu * normal_force;
  constraints[friction_idx] = friction;
}
`;

export const MATH_UTILS_WGSL = `// ─── AVBD Math Utilities for WGSL ───────────────────────────────────────────
// Provides matrix operations for 3x3 (2D) and 6x6 (3D) local solves.

// ─── 3x3 LDL^T Solver (for 2D mode: 3-DOF per body) ────────────────────────

// Solve A * x = b where A is a 3x3 SPD matrix.
// A is stored as mat3x3<f32> (column-major).
// Returns the solution x as a vec3<f32>.
fn solve_ldl3(A: mat3x3<f32>, b: vec3<f32>) -> vec3<f32> {
  // LDL^T decomposition
  // Column 0
  let D0 = A[0][0];
  let L10 = A[0][1] / D0;
  let L20 = A[0][2] / D0;

  // Column 1
  let D1 = A[1][1] - L10 * L10 * D0;
  let L21 = (A[1][2] - L20 * L10 * D0) / D1;

  // Column 2
  let D2 = A[2][2] - L20 * L20 * D0 - L21 * L21 * D1;

  // Forward substitution: L * y = b
  let y0 = b.x;
  let y1 = b.y - L10 * y0;
  let y2 = b.z - L20 * y0 - L21 * y1;

  // Diagonal solve: D * z = y
  let z0 = y0 / D0;
  let z1 = y1 / D1;
  let z2 = y2 / D2;

  // Back substitution: L^T * x = z
  let x2 = z2;
  let x1 = z1 - L21 * x2;
  let x0 = z0 - L10 * x1 - L20 * x2;

  return vec3<f32>(x0, x1, x2);
}

// Outer product of two vec3: result = a * b^T
fn outer_product3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(
    a * b.x,  // column 0
    a * b.y,  // column 1
    a * b.z,  // column 2
  );
}

// Diagonal matrix from absolute values of a vec3 scaled by a scalar
fn diag_abs_scaled(v: vec3<f32>, s: f32) -> mat3x3<f32> {
  return mat3x3<f32>(
    vec3<f32>(abs(v.x) * s, 0.0, 0.0),
    vec3<f32>(0.0, abs(v.y) * s, 0.0),
    vec3<f32>(0.0, 0.0, abs(v.z) * s),
  );
}

// ─── 6x6 LDL^T Solver (for 3D mode: 6-DOF per body) ────────────────────────
// Stored as array<f32, 36> in column-major order.

fn mat6_get(M: array<f32, 36>, row: u32, col: u32) -> f32 {
  return M[col * 6u + row];
}

fn solve_ldl6(A: array<f32, 36>, b: array<f32, 6>) -> array<f32, 6> {
  // L is unit lower triangular, stored below diagonal
  var L: array<f32, 36>;
  var D: array<f32, 6>;

  // Forward pass: compute L and D
  for (var j = 0u; j < 6u; j++) {
    // Compute D[j]
    var sum_d = mat6_get(A, j, j);
    for (var k = 0u; k < j; k++) {
      let ljk = L[k * 6u + j];
      sum_d -= ljk * ljk * D[k];
    }
    D[j] = sum_d;

    // Compute L[i][j] for i > j
    for (var i = j + 1u; i < 6u; i++) {
      var sum_l = mat6_get(A, i, j);
      for (var k = 0u; k < j; k++) {
        sum_l -= L[k * 6u + i] * L[k * 6u + j] * D[k];
      }
      L[j * 6u + i] = sum_l / D[j];
    }
  }

  // Forward substitution: L * y = b
  var y: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) {
    var sum_y = b[i];
    for (var k = 0u; k < i; k++) {
      sum_y -= L[k * 6u + i] * y[k];
    }
    y[i] = sum_y;
  }

  // Diagonal solve: D * z = y
  var z: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) {
    z[i] = y[i] / D[i];
  }

  // Back substitution: L^T * x = z
  var x: array<f32, 6>;
  for (var ii = 0u; ii < 6u; ii++) {
    let i = 5u - ii;
    var sum_x = z[i];
    for (var k = i + 1u; k < 6u; k++) {
      sum_x -= L[i * 6u + k] * x[k];
    }
    x[i] = sum_x;
  }

  return x;
}
`;

export const PRIMAL_UPDATE_2D_WGSL = `// ─── AVBD 2D Primal Update Compute Shader ───────────────────────────────────
// Processes one color group of bodies per dispatch.
// Each thread handles one body: builds 3x3 SPD system, solves via LDL^T.
// Bodies in the same color group share NO constraints → safe parallel update.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
}

struct ConstraintRow {
  body_a: i32,
  body_b: i32,
  force_type: u32,
  _pad0: u32,
  jacobian_a: vec4<f32>,
  jacobian_b: vec4<f32>,
  hessian_diag_a: vec4<f32>,
  hessian_diag_b: vec4<f32>,
  c: f32,
  c0: f32,
  lambda: f32,
  penalty: f32,
  stiffness: f32,
  fmin: f32,
  fmax: f32,
  is_active: u32,
}

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read_write> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read> constraints: array<ConstraintRow>;
@group(0) @binding(4) var<storage, read> color_body_indices: array<u32>;
@group(0) @binding(5) var<storage, read> body_constraint_ranges: array<u32>;
@group(0) @binding(6) var<storage, read> constraint_indices: array<u32>;

fn solve_ldl3(A: mat3x3<f32>, b: vec3<f32>) -> vec3<f32> {
  // Regularize diagonal to prevent f32 catastrophic cancellation.
  // When penalty >> mass/dt², off-diagonal J*J^T*penalty terms can
  // exactly cancel diagonal terms (e.g. 4e9 + 77.76 - 4e9 = 0 in f32).
  // Adding eps proportional to max diagonal keeps the matrix SPD.
  let max_diag = max(A[0][0], max(A[1][1], A[2][2]));
  let eps = 1e-6 * max_diag;

  let D0 = A[0][0] + eps;
  if (D0 <= 0.0) { return vec3<f32>(0.0); }
  let L10 = A[0][1] / D0;
  let L20 = A[0][2] / D0;
  let D1 = A[1][1] + eps - L10 * L10 * D0;
  if (D1 <= 0.0) { return vec3<f32>(0.0); }
  let L21 = (A[1][2] - L20 * L10 * D0) / D1;
  let D2 = A[2][2] + eps - L20 * L20 * D0 - L21 * L21 * D1;
  if (D2 <= 0.0) { return vec3<f32>(0.0); }

  let y0 = b.x;
  let y1 = b.y - L10 * y0;
  let y2 = b.z - L20 * y0 - L21 * y1;

  let x2 = y2 / D2;
  let x1 = y1 / D1 - L21 * x2;
  let x0 = y0 / D0 - L10 * x1 - L20 * x2;

  return vec3<f32>(x0, x1, x2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let thread_id = gid.x;

  // Use num_bodies_in_group from uniform (not arrayLength which returns buffer capacity)
  if (thread_id >= params.num_bodies_in_group) { return; }

  let body_idx = color_body_indices[thread_id];
  let dt = params.dt;
  let dt2 = dt * dt;

  let base = body_idx * 8u;
  let x = body_state[base + 0u];
  let y = body_state[base + 1u];
  let angle = body_state[base + 2u];
  let mass = body_state[base + 6u];
  let inertia = body_state[base + 7u];

  if (mass <= 0.0) { return; }

  let prev_base = body_idx * 8u;
  let prev_x = body_prev[prev_base + 0u];
  let prev_y = body_prev[prev_base + 1u];
  let prev_angle = body_prev[prev_base + 2u];
  let inertial_x = body_prev[prev_base + 3u];
  let inertial_y = body_prev[prev_base + 4u];
  let inertial_angle = body_prev[prev_base + 5u];

  var lhs = mat3x3<f32>(
    vec3<f32>(mass / dt2, 0.0, 0.0),
    vec3<f32>(0.0, mass / dt2, 0.0),
    vec3<f32>(0.0, 0.0, inertia / dt2),
  );

  var rhs = vec3<f32>(
    mass / dt2 * (x - inertial_x),
    mass / dt2 * (y - inertial_y),
    inertia / dt2 * (angle - inertial_angle),
  );

  let range_base = body_idx * 2u;
  let constraint_start = body_constraint_ranges[range_base + 0u];
  let constraint_count = body_constraint_ranges[range_base + 1u];

  for (var ci = 0u; ci < constraint_count; ci++) {
    let cr_idx = constraint_indices[constraint_start + ci];
    let cr = constraints[cr_idx];

    if (cr.is_active == 0u) { continue; }

    var J: vec3<f32>;
    var H_diag: vec3<f32>;
    if (cr.body_a == i32(body_idx)) {
      J = cr.jacobian_a.xyz;
      H_diag = cr.hessian_diag_a.xyz;
    } else {
      J = cr.jacobian_b.xyz;
      H_diag = cr.hessian_diag_b.xyz;
    }

    // Evaluate linearized constraint: C = C0 + J_A·dp_A + J_B·dp_B
    var c_eval = cr.c0;
    if (cr.body_a >= 0) {
      let ba_base = u32(cr.body_a) * 8u;
      c_eval += cr.jacobian_a.x * (body_state[ba_base + 0u] - body_prev[ba_base + 0u])
             + cr.jacobian_a.y * (body_state[ba_base + 1u] - body_prev[ba_base + 1u])
             + cr.jacobian_a.z * (body_state[ba_base + 2u] - body_prev[ba_base + 2u]);
    }
    if (cr.body_b >= 0) {
      let bb_base = u32(cr.body_b) * 8u;
      c_eval += cr.jacobian_b.x * (body_state[bb_base + 0u] - body_prev[bb_base + 0u])
             + cr.jacobian_b.y * (body_state[bb_base + 1u] - body_prev[bb_base + 1u])
             + cr.jacobian_b.z * (body_state[bb_base + 2u] - body_prev[bb_base + 2u]);
    }

    // Stiffness guard: soft constraints (finite stiffness) use lambda=0 in primal
    var lambda_for_primal = cr.lambda;
    if (cr.stiffness < 1e30) {
      lambda_for_primal = 0.0;
    }

    var f = cr.penalty * c_eval + lambda_for_primal;
    f = clamp(f, cr.fmin, cr.fmax);

    rhs += J * f;

    let col0 = J * J.x * cr.penalty;
    let col1 = J * J.y * cr.penalty;
    let col2 = J * J.z * cr.penalty;
    lhs[0] += col0;
    lhs[1] += col1;
    lhs[2] += col2;

    let abs_f = abs(f);
    lhs[0].x += abs(H_diag.x) * abs_f;
    lhs[1].y += abs(H_diag.y) * abs_f;
    lhs[2].z += abs(H_diag.z) * abs_f;
  }

  if (lhs[0].x <= 0.0 || lhs[1].y <= 0.0 || lhs[2].z <= 0.0) { return; }

  let delta = solve_ldl3(lhs, rhs);

  body_state[base + 0u] = x - delta.x;
  body_state[base + 1u] = y - delta.y;
  body_state[base + 2u] = angle - delta.z;
}
`;

export const PRIMAL_UPDATE_3D_WGSL = `// ─── AVBD 3D Primal Update Compute Shader ───────────────────────────────────
// 6-DOF per body: position (x,y,z) + angular correction (wx,wy,wz)
// Builds a 6x6 SPD system and solves via LDL^T.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  gravity_z: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
}

struct ConstraintRow3D {
  body_a: i32,
  body_b: i32,
  force_type: u32,
  _pad0: u32,
  jacobian_a_lin: vec4<f32>,
  jacobian_a_ang: vec4<f32>,
  jacobian_b_lin: vec4<f32>,
  jacobian_b_ang: vec4<f32>,
  c: f32,
  c0: f32,
  lambda: f32,
  penalty: f32,
  stiffness: f32,
  fmin: f32,
  fmax: f32,
  is_active: u32,
}

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read_write> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read> constraints: array<ConstraintRow3D>;
@group(0) @binding(4) var<storage, read> color_body_indices: array<u32>;
@group(0) @binding(5) var<storage, read> body_constraint_ranges: array<u32>;
@group(0) @binding(6) var<storage, read> constraint_indices: array<u32>;

// 6x6 LDL^T solver
fn solve_ldl6(A: array<f32, 36>, b: array<f32, 6>) -> array<f32, 6> {
  var L: array<f32, 36>;
  var D: array<f32, 6>;

  // Regularize diagonal to prevent f32 catastrophic cancellation
  var max_diag: f32 = 0.0;
  for (var i = 0u; i < 6u; i++) {
    max_diag = max(max_diag, A[i * 6u + i]);
  }
  let eps6 = 1e-6 * max_diag;

  for (var j = 0u; j < 6u; j++) {
    var sum_d = A[j * 6u + j] + eps6;
    for (var k = 0u; k < j; k++) {
      let ljk = L[k * 6u + j];
      sum_d -= ljk * ljk * D[k];
    }
    D[j] = sum_d;
    if (D[j] <= 0.0) {
      var zero: array<f32, 6>;
      return zero;
    }

    for (var i = j + 1u; i < 6u; i++) {
      var sum_l = A[j * 6u + i];
      for (var k = 0u; k < j; k++) {
        sum_l -= L[k * 6u + i] * L[k * 6u + j] * D[k];
      }
      L[j * 6u + i] = sum_l / D[j];
    }
  }

  var y: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) {
    var s = b[i];
    for (var k = 0u; k < i; k++) { s -= L[k * 6u + i] * y[k]; }
    y[i] = s;
  }

  var z: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) { z[i] = y[i] / D[i]; }

  var x: array<f32, 6>;
  for (var ii = 0u; ii < 6u; ii++) {
    let i = 5u - ii;
    var s = z[i];
    for (var k = i + 1u; k < 6u; k++) { s -= L[i * 6u + k] * x[k]; }
    x[i] = s;
  }
  return x;
}

// Body state 3D layout (20 floats per body):
// [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, mass, Ix, Iy, Iz, pad, pad, pad]
// Body prev 3D layout (14 floats per body):
// [px, py, pz, pqw, pqx, pqy, pqz, ix, iy, iz, iqw, iqx, iqy, iqz]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let thread_id = gid.x;
  if (thread_id >= params.num_bodies_in_group) { return; }

  let body_idx = color_body_indices[thread_id];
  let dt = params.dt;
  let dt2 = dt * dt;

  let base = body_idx * 20u;
  let mass = body_state[base + 13u];
  if (mass <= 0.0) { return; }

  let Ix = body_state[base + 14u];
  let Iy = body_state[base + 15u];
  let Iz = body_state[base + 16u];

  let x = body_state[base + 0u];
  let y = body_state[base + 1u];
  let z = body_state[base + 2u];

  let prev_base = body_idx * 14u;
  let prev_x = body_prev[prev_base + 0u];
  let prev_y = body_prev[prev_base + 1u];
  let prev_z = body_prev[prev_base + 2u];
  let inertial_x = body_prev[prev_base + 7u];
  let inertial_y = body_prev[prev_base + 8u];
  let inertial_z = body_prev[prev_base + 9u];

  // Initialize 6x6 LHS diagonal
  var lhs: array<f32, 36>;
  for (var i = 0u; i < 36u; i++) { lhs[i] = 0.0; }
  lhs[0 * 6u + 0u] = mass / dt2;
  lhs[1 * 6u + 1u] = mass / dt2;
  lhs[2 * 6u + 2u] = mass / dt2;
  lhs[3 * 6u + 3u] = Ix / dt2;
  lhs[4 * 6u + 4u] = Iy / dt2;
  lhs[5 * 6u + 5u] = Iz / dt2;

  var rhs: array<f32, 6>;
  rhs[0] = mass / dt2 * (x - inertial_x);
  rhs[1] = mass / dt2 * (y - inertial_y);
  rhs[2] = mass / dt2 * (z - inertial_z);
  rhs[3] = 0.0;
  rhs[4] = 0.0;
  rhs[5] = 0.0;

  // Accumulate constraint contributions
  let range_base = body_idx * 2u;
  let constraint_start = body_constraint_ranges[range_base + 0u];
  let constraint_count = body_constraint_ranges[range_base + 1u];

  for (var ci = 0u; ci < constraint_count; ci++) {
    let cr_idx = constraint_indices[constraint_start + ci];
    let cr = constraints[cr_idx];
    if (cr.is_active == 0u) { continue; }

    var J: array<f32, 6>;
    if (cr.body_a == i32(body_idx)) {
      J[0] = cr.jacobian_a_lin.x; J[1] = cr.jacobian_a_lin.y; J[2] = cr.jacobian_a_lin.z;
      J[3] = cr.jacobian_a_ang.x; J[4] = cr.jacobian_a_ang.y; J[5] = cr.jacobian_a_ang.z;
    } else {
      J[0] = cr.jacobian_b_lin.x; J[1] = cr.jacobian_b_lin.y; J[2] = cr.jacobian_b_lin.z;
      J[3] = cr.jacobian_b_ang.x; J[4] = cr.jacobian_b_ang.y; J[5] = cr.jacobian_b_ang.z;
    }

    // Full Taylor-series constraint evaluation: C = C0 + J_A·dp_A + J_B·dp_B
    var c_eval = cr.c0;
    if (cr.body_a >= 0) {
      let ba = u32(cr.body_a) * 20u;
      let bap = u32(cr.body_a) * 14u;
      let dpx = body_state[ba + 0u] - body_prev[bap + 0u];
      let dpy = body_state[ba + 1u] - body_prev[bap + 1u];
      let dpz = body_state[ba + 2u] - body_prev[bap + 2u];
      c_eval += cr.jacobian_a_lin.x * dpx + cr.jacobian_a_lin.y * dpy + cr.jacobian_a_lin.z * dpz;
      // Angular displacement via quaternion difference: dq = q * conj(q_prev)
      let qw_a = body_state[ba + 3u]; let qx_a = body_state[ba + 4u];
      let qy_a = body_state[ba + 5u]; let qz_a = body_state[ba + 6u];
      let pqw_a = body_prev[bap + 3u]; let pqx_a = body_prev[bap + 4u];
      let pqy_a = body_prev[bap + 5u]; let pqz_a = body_prev[bap + 6u];
      let dqx_a = pqw_a * qx_a - pqx_a * qw_a - pqy_a * qz_a + pqz_a * qy_a;
      let dqy_a = pqw_a * qy_a + pqx_a * qz_a - pqy_a * qw_a - pqz_a * qx_a;
      let dqz_a = pqw_a * qz_a - pqx_a * qy_a + pqy_a * qx_a - pqz_a * qw_a;
      c_eval += cr.jacobian_a_ang.x * 2.0 * dqx_a
             + cr.jacobian_a_ang.y * 2.0 * dqy_a
             + cr.jacobian_a_ang.z * 2.0 * dqz_a;
    }
    if (cr.body_b >= 0) {
      let bb = u32(cr.body_b) * 20u;
      let bbp = u32(cr.body_b) * 14u;
      let dpx = body_state[bb + 0u] - body_prev[bbp + 0u];
      let dpy = body_state[bb + 1u] - body_prev[bbp + 1u];
      let dpz = body_state[bb + 2u] - body_prev[bbp + 2u];
      c_eval += cr.jacobian_b_lin.x * dpx + cr.jacobian_b_lin.y * dpy + cr.jacobian_b_lin.z * dpz;
      let qw_b = body_state[bb + 3u]; let qx_b = body_state[bb + 4u];
      let qy_b = body_state[bb + 5u]; let qz_b = body_state[bb + 6u];
      let pqw_b = body_prev[bbp + 3u]; let pqx_b = body_prev[bbp + 4u];
      let pqy_b = body_prev[bbp + 5u]; let pqz_b = body_prev[bbp + 6u];
      let dqx_b = pqw_b * qx_b - pqx_b * qw_b - pqy_b * qz_b + pqz_b * qy_b;
      let dqy_b = pqw_b * qy_b + pqx_b * qz_b - pqy_b * qw_b - pqz_b * qx_b;
      let dqz_b = pqw_b * qz_b - pqx_b * qy_b + pqy_b * qx_b - pqz_b * qw_b;
      c_eval += cr.jacobian_b_ang.x * 2.0 * dqx_b
             + cr.jacobian_b_ang.y * 2.0 * dqy_b
             + cr.jacobian_b_ang.z * 2.0 * dqz_b;
    }

    // Stiffness guard
    var lambda_for_primal = cr.lambda;
    if (cr.stiffness < 1e30) { lambda_for_primal = 0.0; }

    var f = cr.penalty * c_eval + lambda_for_primal;
    f = clamp(f, cr.fmin, cr.fmax);

    for (var k = 0u; k < 6u; k++) { rhs[k] += J[k] * f; }
    for (var i = 0u; i < 6u; i++) {
      for (var j = 0u; j < 6u; j++) {
        lhs[j * 6u + i] += J[i] * J[j] * cr.penalty;
      }
    }
  }

  // Check diagonal validity
  for (var i = 0u; i < 6u; i++) {
    if (lhs[i * 6u + i] <= 0.0) { return; }
  }

  let delta = solve_ldl6(lhs, rhs);

  body_state[base + 0u] = x - delta[0];
  body_state[base + 1u] = y - delta[1];
  body_state[base + 2u] = z - delta[2];

  // Quaternion update from angular correction
  let qw = body_state[base + 3u];
  let qx = body_state[base + 4u];
  let qy = body_state[base + 5u];
  let qz = body_state[base + 6u];

  let hw = -0.5 * (delta[3] * qx + delta[4] * qy + delta[5] * qz);
  let hx =  0.5 * (delta[3] * qw + delta[5] * qy - delta[4] * qz);
  let hy =  0.5 * (delta[4] * qw + delta[3] * qz - delta[5] * qx);
  let hz =  0.5 * (delta[5] * qw + delta[4] * qx - delta[3] * qy);

  var nqw = qw - hw;
  var nqx = qx - hx;
  var nqy = qy - hy;
  var nqz = qz - hz;

  let qlen = sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz);
  if (qlen > 0.0) {
    nqw /= qlen; nqx /= qlen; nqy /= qlen; nqz /= qlen;
  }

  body_state[base + 3u] = nqw;
  body_state[base + 4u] = nqx;
  body_state[base + 5u] = nqy;
  body_state[base + 6u] = nqz;
}
`;

