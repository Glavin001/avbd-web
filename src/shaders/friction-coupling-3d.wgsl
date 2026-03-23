// ─── AVBD 3D Friction Coupling Compute Shader ───────────────────────────────
// Runs as a separate compute pass AFTER the 3D dual update.
// 3D contact rows are ordered as triplets: [normal, friction1, friction2, ...].
// One thread per contact TRIPLET (not per row).
// Updates both friction tangent bounds based on the finalized normal lambda.

struct FrictionParams {
  num_constraints: u32,
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

@group(0) @binding(0) var<uniform> params: FrictionParams;
@group(0) @binding(1) var<storage, read_write> constraints: array<ConstraintRow3D>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let triplet_idx = gid.x;
  let normal_idx = triplet_idx * 3u;
  let friction1_idx = normal_idx + 1u;
  let friction2_idx = normal_idx + 2u;

  if (friction2_idx >= params.num_constraints) { return; }

  let normal = constraints[normal_idx];
  // Only process contact triplets (force_type 0 = Contact)
  if (normal.force_type != 0u || normal.is_active == 0u) { return; }

  // Coulomb friction: |f_tangent| <= mu * |f_normal|
  // mu is packed in jacobian_b_ang.w of the normal row during CPU upload
  let mu = normal.jacobian_b_ang.w;
  let normal_force = abs(normal.lambda);

  // Update friction tangent 1
  var fric1 = constraints[friction1_idx];
  if (fric1.force_type == 0u && fric1.is_active != 0u) {
    fric1.fmin = -mu * normal_force;
    fric1.fmax = mu * normal_force;
    constraints[friction1_idx] = fric1;
  }

  // Update friction tangent 2
  var fric2 = constraints[friction2_idx];
  if (fric2.force_type == 0u && fric2.is_active != 0u) {
    fric2.fmin = -mu * normal_force;
    fric2.fmax = mu * normal_force;
    constraints[friction2_idx] = fric2;
  }
}
