// ─── AVBD Dual Update Compute Shader ────────────────────────────────────────
// Updates lambda (Lagrange multiplier) and ramps penalty for each constraint.
// Shared between 2D and 3D modes.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  penalty_min: f32,
  penalty_max: f32,
  num_bodies: u32,
  num_constraints: u32,
  beta: f32,   // penalty growth rate
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
  active: u32,
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
  if (cr.active == 0u) { return; }

  // Re-evaluate linearized constraint
  var c_eval = cr.c0;
  if (cr.body_a >= 0) {
    let ba_base = u32(cr.body_a) * 8u;
    let ba_prev_base = u32(cr.body_a) * 8u;
    c_eval += cr.jacobian_a.x * (body_state[ba_base + 0u] - body_prev[ba_prev_base + 0u])
           + cr.jacobian_a.y * (body_state[ba_base + 1u] - body_prev[ba_prev_base + 1u])
           + cr.jacobian_a.z * (body_state[ba_base + 2u] - body_prev[ba_prev_base + 2u]);
  }
  if (cr.body_b >= 0) {
    let bb_base = u32(cr.body_b) * 8u;
    let bb_prev_base = u32(cr.body_b) * 8u;
    c_eval += cr.jacobian_b.x * (body_state[bb_base + 0u] - body_prev[bb_prev_base + 0u])
           + cr.jacobian_b.y * (body_state[bb_base + 1u] - body_prev[bb_prev_base + 1u])
           + cr.jacobian_b.z * (body_state[bb_base + 2u] - body_prev[bb_prev_base + 2u]);
  }

  // Update lambda
  var new_lambda = cr.penalty * c_eval + cr.lambda;
  new_lambda = clamp(new_lambda, cr.fmin, cr.fmax);
  cr.lambda = new_lambda;

  // Ramp penalty
  cr.penalty += params.beta * abs(c_eval);
  cr.penalty = clamp(cr.penalty, params.penalty_min, params.penalty_max);
  if (cr.penalty > cr.stiffness) {
    cr.penalty = cr.stiffness;
  }

  // Write back
  constraints[idx] = cr;
}
