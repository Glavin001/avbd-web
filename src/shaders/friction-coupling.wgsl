// ─── AVBD Friction Coupling Compute Shader ──────────────────────────────────
// Runs as a separate compute pass AFTER the dual update.
// Per the AVBD reference (manifold.cpp: computeConstraint), friction bounds
// are updated using the current normal lambda BETWEEN iterations.
// Running after dual guarantees all normal lambdas are finalized.
// Contact rows are ordered [normal, friction, normal, friction, ...].
// One thread per contact PAIR (not per row).

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

  // Coulomb friction: |f_tangent| <= mu * |f_normal|
  // mu is packed in hessian_diag_b.w of the normal row during CPU upload
  let mu = normal.hessian_diag_b.w;
  let normal_force = abs(normal.lambda);
  friction.fmin = -mu * normal_force;
  friction.fmax = mu * normal_force;
  constraints[friction_idx] = friction;
}
