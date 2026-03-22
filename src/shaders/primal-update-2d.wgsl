// ─── AVBD 2D Primal Update Compute Shader ───────────────────────────────────
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

fn solve_ldl3(A: mat3x3<f32>, b: vec3<f32>) -> vec3<f32> {
  let D0 = A[0][0];
  if (D0 <= 0.0) { return vec3<f32>(0.0); }
  let L10 = A[0][1] / D0;
  let L20 = A[0][2] / D0;
  let D1 = A[1][1] - L10 * L10 * D0;
  if (D1 <= 0.0) { return vec3<f32>(0.0); }
  let L21 = (A[1][2] - L20 * L10 * D0) / D1;
  let D2 = A[2][2] - L20 * L20 * D0 - L21 * L21 * D1;
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
    let cr_idx = constraint_start + ci;
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
