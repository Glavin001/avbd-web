// ─── AVBD 3D Primal Update Compute Shader ───────────────────────────────────
// 6-DOF per body: [x, y, z, wx, wy, wz]
// Builds a 6x6 SPD system and solves via LDL^T.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  gravity_z: f32,
  penalty_min: f32,
  penalty_max: f32,
  num_bodies: u32,
  num_constraints: u32,
  is_stabilization: u32,
}

struct ConstraintRow3D {
  body_a: i32,
  body_b: i32,
  force_type: u32,
  _pad0: u32,
  // Jacobians stored as 2 x vec4 (8 floats for 6 DOF + padding)
  jacobian_a_lin: vec4<f32>,  // [Jax, Jay, Jaz, 0]
  jacobian_a_ang: vec4<f32>,  // [Jawx, Jawy, Jawz, 0]
  jacobian_b_lin: vec4<f32>,
  jacobian_b_ang: vec4<f32>,
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
@group(0) @binding(1) var<storage, read_write> body_state: array<f32>;
// Body state 3D: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, mass, Ix, Iy, Iz, pad, pad, pad]
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
// prev: [px, py, pz, pqw, pqx, pqy, pqz, ix, iy, iz, iqw, iqx, iqy, iqz, pad...]
@group(0) @binding(3) var<storage, read> constraints: array<ConstraintRow3D>;
@group(0) @binding(4) var<storage, read> color_body_indices: array<u32>;
@group(0) @binding(5) var<storage, read> body_constraint_ranges: array<u32>;

// ─── 6x6 LDL^T Solver ──────────────────────────────────────────────────────

fn solve_ldl6(A: array<f32, 36>, b: array<f32, 6>) -> array<f32, 6> {
  var L: array<f32, 36>;
  var D: array<f32, 6>;

  for (var j = 0u; j < 6u; j++) {
    var sum_d = A[j * 6u + j];
    for (var k = 0u; k < j; k++) {
      let ljk = L[k * 6u + j];
      sum_d -= ljk * ljk * D[k];
    }
    D[j] = sum_d;

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
    for (var k = 0u; k < i; k++) {
      s -= L[k * 6u + i] * y[k];
    }
    y[i] = s;
  }

  var z: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) {
    z[i] = y[i] / D[i];
  }

  var x: array<f32, 6>;
  for (var ii = 0u; ii < 6u; ii++) {
    let i = 5u - ii;
    var s = z[i];
    for (var k = i + 1u; k < 6u; k++) {
      s -= L[i * 6u + k] * x[k];
    }
    x[i] = s;
  }

  return x;
}

// ─── Main ───────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let thread_id = gid.x;
  let num_bodies_in_group = arrayLength(&color_body_indices);
  if (thread_id >= num_bodies_in_group) { return; }

  let body_idx = color_body_indices[thread_id];
  let dt = params.dt;
  let dt2 = dt * dt;

  // Read body mass/inertia from state buffer
  let base = body_idx * 20u;
  let mass = body_state[base + 13u];
  if (mass <= 0.0) { return; } // Fixed body

  let Ix = body_state[base + 14u];
  let Iy = body_state[base + 15u];
  let Iz = body_state[base + 16u];

  // Current position
  let x = body_state[base + 0u];
  let y = body_state[base + 1u];
  let z = body_state[base + 2u];

  // Read prev/inertial state
  let prev_base = body_idx * 14u;
  let prev_x = body_prev[prev_base + 0u];
  let prev_y = body_prev[prev_base + 1u];
  let prev_z = body_prev[prev_base + 2u];
  let inertial_x = body_prev[prev_base + 7u];
  let inertial_y = body_prev[prev_base + 8u];
  let inertial_z = body_prev[prev_base + 9u];

  // Initialize 6x6 LHS (diagonal mass matrix / dt^2)
  var lhs: array<f32, 36>;
  for (var i = 0u; i < 36u; i++) { lhs[i] = 0.0; }
  lhs[0 * 6u + 0u] = mass / dt2;
  lhs[1 * 6u + 1u] = mass / dt2;
  lhs[2 * 6u + 2u] = mass / dt2;
  lhs[3 * 6u + 3u] = Ix / dt2;
  lhs[4 * 6u + 4u] = Iy / dt2;
  lhs[5 * 6u + 5u] = Iz / dt2;

  // RHS = M/dt^2 * (x - x_inertial)
  // Note: rotation error would need quaternion diff, simplified here
  var rhs: array<f32, 6>;
  rhs[0] = mass / dt2 * (x - inertial_x);
  rhs[1] = mass / dt2 * (y - inertial_y);
  rhs[2] = mass / dt2 * (z - inertial_z);
  rhs[3] = 0.0; // Rotation error (would come from quaternion diff)
  rhs[4] = 0.0;
  rhs[5] = 0.0;

  // Accumulate constraint contributions
  let range_base = body_idx * 2u;
  let constraint_start = body_constraint_ranges[range_base + 0u];
  let constraint_count = body_constraint_ranges[range_base + 1u];

  for (var ci = 0u; ci < constraint_count; ci++) {
    let cr_idx = constraint_start + ci;
    let cr = constraints[cr_idx];
    if (cr.active == 0u) { continue; }

    var J: array<f32, 6>;
    if (cr.body_a == i32(body_idx)) {
      J[0] = cr.jacobian_a_lin.x;
      J[1] = cr.jacobian_a_lin.y;
      J[2] = cr.jacobian_a_lin.z;
      J[3] = cr.jacobian_a_ang.x;
      J[4] = cr.jacobian_a_ang.y;
      J[5] = cr.jacobian_a_ang.z;
    } else {
      J[0] = cr.jacobian_b_lin.x;
      J[1] = cr.jacobian_b_lin.y;
      J[2] = cr.jacobian_b_lin.z;
      J[3] = cr.jacobian_b_ang.x;
      J[4] = cr.jacobian_b_ang.y;
      J[5] = cr.jacobian_b_ang.z;
    }

    // Evaluate linearized constraint (simplified — full version reads from body buffers)
    let c_eval = cr.c0; // Simplified: would add J * dp

    var f = cr.penalty * c_eval + cr.lambda;
    f = clamp(f, cr.fmin, cr.fmax);

    // Accumulate RHS and LHS
    for (var k = 0u; k < 6u; k++) {
      rhs[k] += J[k] * f;
    }
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

  // Solve
  let delta = solve_ldl6(lhs, rhs);

  // Apply position correction
  body_state[base + 0u] = x - delta[0];
  body_state[base + 1u] = y - delta[1];
  body_state[base + 2u] = z - delta[2];

  // Apply rotation correction via quaternion update
  // delta[3..5] are angular corrections
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

  // Normalize quaternion
  let qlen = sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz);
  if (qlen > 0.0) {
    nqw /= qlen; nqx /= qlen; nqy /= qlen; nqz /= qlen;
  }

  body_state[base + 3u] = nqw;
  body_state[base + 4u] = nqx;
  body_state[base + 5u] = nqy;
  body_state[base + 6u] = nqz;
}
