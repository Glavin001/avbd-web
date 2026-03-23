// ─── AVBD 3D Dual Update Compute Shader ────────────────────────────────────
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
  hessian_diag_a_ang: vec4<f32>,
  hessian_diag_b_ang: vec4<f32>,
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

  // Conditional penalty ramp: only when constraint is interior.
  // Skip ramping for friction rows (Contact type with finite fmin) — friction penalty
  // should stay low to avoid stiff angular springs that cause spinning instability.
  // Normal contact rows have fmin=-inf; friction rows have finite Coulomb bounds.
  let is_friction = cr.force_type == 0u && cr.fmin > -1e30;
  if (!is_friction && cr.lambda > cr.fmin && cr.lambda < cr.fmax) {
    cr.penalty += params.beta * abs(c_eval);
  }
  cr.penalty = clamp(cr.penalty, params.penalty_min, params.penalty_max);
  if (cr.penalty > cr.stiffness) {
    cr.penalty = cr.stiffness;
  }

  // Write back this row
  constraints[idx] = cr;

  // NOTE: Friction coupling runs as a SEPARATE compute pass (friction-coupling.wgsl)
  // after the dual update, guaranteeing all normal lambdas are finalized.
}
