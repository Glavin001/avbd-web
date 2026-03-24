// ─── 3D Constraint Assembly ──────────────────────────────────────────────────
// Converts 3D contact points into solver constraint rows.
// Each contact generates 3 rows: normal + 2 friction tangents.
//
// Contact input format (12 u32s per contact):
//   [bodyA, bodyB, featureId, _pad,
//    normal_x, normal_y, normal_z, depth,
//    point_x, point_y, point_z, mu]
//
// Constraint output format (36 floats per row, matching existing 3D primal/dual):
//   [bodyA(i32), bodyB(i32), forceType(u32), _pad(u32),
//    jacobian_a_lin(vec4), jacobian_a_ang(vec4),
//    jacobian_b_lin(vec4), jacobian_b_ang(vec4),
//    hessian_diag_a_ang(vec4), hessian_diag_b_ang(vec4),
//    c, c0, lambda, penalty, stiffness, fmin, fmax, active(u32)]

struct AssemblyParams3D {
  max_contacts: u32,
  max_constraints: u32,
  collision_margin: f32,
  dt: f32,
  penalty_min: f32,
  alpha: f32,
  warmstart_cache_size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: AssemblyParams3D;
@group(0) @binding(1) var<storage, read> contact_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> body_state: array<f32>;      // 20 floats/body
@group(0) @binding(3) var<storage, read> collider_info: array<u32>;   // 8 u32s/body
@group(0) @binding(4) var<storage, read_write> constraint_buffer: array<f32>;
@group(0) @binding(5) var<storage, read_write> constraint_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> warmstart_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> warmstart_vals: array<f32>;
@group(0) @binding(8) var<storage, read_write> warmstart_age: array<u32>;

const BODY_STRIDE: u32 = 20u;
const CONTACT_STRIDE: u32 = 12u;
const CONSTRAINT_STRIDE: u32 = 36u;
const FORCE_TYPE_CONTACT: u32 = 0u;

fn load_contact_f32(idx: u32, field: u32) -> f32 {
  return bitcast<f32>(contact_buffer[idx * CONTACT_STRIDE + field]);
}

fn load_contact_u32(idx: u32, field: u32) -> u32 {
  return contact_buffer[idx * CONTACT_STRIDE + field];
}

// Duff et al. 2017 — branch-free orthonormal basis from normal
fn build_tangent_frame(n: vec3<f32>) -> mat2x3<f32> {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  let t1 = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
  let t2 = vec3<f32>(b, s + n.y * n.y * a, -n.y);
  return mat2x3<f32>(t1, t2);
}

// Cross product
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x,
  );
}

fn warmstart_hash(bodyA: u32, bodyB: u32, featureId: u32) -> u32 {
  var h = bodyA * 73856093u;
  h = h ^ (bodyB * 19349663u);
  h = h ^ (featureId * 83492791u);
  return h % params.warmstart_cache_size;
}

fn warmstart_lookup(bodyA: u32, bodyB: u32, featureId: u32) -> f32 {
  if (params.warmstart_cache_size == 0u) {
    return params.penalty_min;
  }
  var slot = warmstart_hash(bodyA, bodyB, featureId);
  for (var probe = 0u; probe < 8u; probe++) {
    let idx = slot * 3u;
    if (warmstart_keys[idx] == bodyA &&
        warmstart_keys[idx + 1u] == bodyB &&
        warmstart_keys[idx + 2u] == featureId) {
      if (warmstart_age[slot] < 5u) {
        return max(params.penalty_min, warmstart_vals[slot * 4u]);
      }
    }
    slot = (slot + 1u) % params.warmstart_cache_size;
  }
  return params.penalty_min;
}

fn write_constraint_3d(row_idx: u32,
                       bodyA: i32, bodyB: i32,
                       jA_lin: vec3<f32>, jA_ang: vec3<f32>,
                       jB_lin: vec3<f32>, jB_ang: vec3<f32>,
                       hA_ang: vec3<f32>, hB_ang: vec3<f32>,
                       c_val: f32, c0_val: f32,
                       lambda: f32, penalty: f32,
                       stiffness: f32,
                       fmin_val: f32, fmax_val: f32,
                       mu: f32) {
  let base = row_idx * CONSTRAINT_STRIDE;

  constraint_buffer[base + 0u] = bitcast<f32>(bodyA);
  constraint_buffer[base + 1u] = bitcast<f32>(bodyB);
  constraint_buffer[base + 2u] = bitcast<f32>(FORCE_TYPE_CONTACT);
  constraint_buffer[base + 3u] = 0.0;

  // Jacobian A linear (vec4)
  constraint_buffer[base + 4u] = jA_lin.x;
  constraint_buffer[base + 5u] = jA_lin.y;
  constraint_buffer[base + 6u] = jA_lin.z;
  constraint_buffer[base + 7u] = 0.0;

  // Jacobian A angular (vec4)
  constraint_buffer[base + 8u] = jA_ang.x;
  constraint_buffer[base + 9u] = jA_ang.y;
  constraint_buffer[base + 10u] = jA_ang.z;
  constraint_buffer[base + 11u] = 0.0;

  // Jacobian B linear (vec4)
  constraint_buffer[base + 12u] = jB_lin.x;
  constraint_buffer[base + 13u] = jB_lin.y;
  constraint_buffer[base + 14u] = jB_lin.z;
  constraint_buffer[base + 15u] = 0.0;

  // Jacobian B angular (vec4, .w stores mu for friction coupling)
  constraint_buffer[base + 16u] = jB_ang.x;
  constraint_buffer[base + 17u] = jB_ang.y;
  constraint_buffer[base + 18u] = jB_ang.z;
  constraint_buffer[base + 19u] = mu;

  // Hessian diag A angular (vec4)
  constraint_buffer[base + 20u] = hA_ang.x;
  constraint_buffer[base + 21u] = hA_ang.y;
  constraint_buffer[base + 22u] = hA_ang.z;
  constraint_buffer[base + 23u] = 0.0;

  // Hessian diag B angular (vec4)
  constraint_buffer[base + 24u] = hB_ang.x;
  constraint_buffer[base + 25u] = hB_ang.y;
  constraint_buffer[base + 26u] = hB_ang.z;
  constraint_buffer[base + 27u] = 0.0;

  // Scalar fields
  constraint_buffer[base + 28u] = c_val;
  constraint_buffer[base + 29u] = c0_val;
  constraint_buffer[base + 30u] = lambda;
  constraint_buffer[base + 31u] = penalty;
  constraint_buffer[base + 32u] = stiffness;
  constraint_buffer[base + 33u] = fmin_val;
  constraint_buffer[base + 34u] = fmax_val;
  constraint_buffer[base + 35u] = bitcast<f32>(1u); // active
}

@compute @workgroup_size(256)
fn constraint_assembly_3d(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let contact_idx = gid.x;
  if (contact_idx >= params.max_contacts) {
    return;
  }

  // Check if this contact is valid
  let bodyA_u32 = load_contact_u32(contact_idx, 0u);
  if (bodyA_u32 == 0xFFFFFFFFu) {
    return;
  }

  let bodyA = i32(bodyA_u32);
  let bodyB = i32(load_contact_u32(contact_idx, 1u));
  let featureId = load_contact_u32(contact_idx, 2u);

  let normal = vec3<f32>(
    load_contact_f32(contact_idx, 4u),
    load_contact_f32(contact_idx, 5u),
    load_contact_f32(contact_idx, 6u),
  );
  let depth = load_contact_f32(contact_idx, 7u);
  let contact_point = vec3<f32>(
    load_contact_f32(contact_idx, 8u),
    load_contact_f32(contact_idx, 9u),
    load_contact_f32(contact_idx, 10u),
  );
  let mu = load_contact_f32(contact_idx, 11u);

  // Load body positions
  let posA = vec3<f32>(
    body_state[u32(bodyA) * BODY_STRIDE + 0u],
    body_state[u32(bodyA) * BODY_STRIDE + 1u],
    body_state[u32(bodyA) * BODY_STRIDE + 2u],
  );
  let posB = vec3<f32>(
    body_state[u32(bodyB) * BODY_STRIDE + 0u],
    body_state[u32(bodyB) * BODY_STRIDE + 1u],
    body_state[u32(bodyB) * BODY_STRIDE + 2u],
  );

  // Lever arms
  let rA = contact_point - posA;
  let rB = contact_point - posB;

  // ─── Normal constraint ─────────────────────────────────
  let jA_lin_n = normal;
  let jA_ang_n = cross3(rA, normal);
  let jB_lin_n = -normal;
  let jB_ang_n = -cross3(rB, normal);

  // Hessian diagonal (angular geometric stiffness)
  let hA_ang_n = -rA * dot(rA, normal) / max(dot(rA, rA), 1e-8);
  let hB_ang_n = -rB * dot(rB, normal) / max(dot(rB, rB), 1e-8);

  let c_normal = -depth + params.collision_margin;

  let penalty = warmstart_lookup(bodyA_u32, u32(bodyB), featureId);

  // Allocate 3 rows atomically
  let row_base = atomicAdd(&constraint_count[0], 3u);
  if (row_base + 3u > params.max_constraints) {
    return;
  }

  // Write normal row
  write_constraint_3d(
    row_base,
    bodyA, bodyB,
    jA_lin_n, jA_ang_n,
    jB_lin_n, jB_ang_n,
    hA_ang_n, hB_ang_n,
    c_normal, c_normal,
    0.0, penalty,
    1e30,
    -1e30, 0.0,
    mu,
  );

  // ─── Friction tangent frame (Duff et al.) ──────────────
  let frame = build_tangent_frame(normal);
  let t1 = frame[0];
  let t2 = frame[1];

  // Friction tangent 1
  let jA_lin_t1 = t1;
  let jA_ang_t1 = cross3(rA, t1);
  let jB_lin_t1 = -t1;
  let jB_ang_t1 = -cross3(rB, t1);
  let hA_ang_t1 = -rA * dot(rA, t1) / max(dot(rA, rA), 1e-8);
  let hB_ang_t1 = -rB * dot(rB, t1) / max(dot(rB, rB), 1e-8);

  let est_force = penalty * depth;
  let fric_limit = mu * est_force;

  write_constraint_3d(
    row_base + 1u,
    bodyA, bodyB,
    jA_lin_t1, jA_ang_t1,
    jB_lin_t1, jB_ang_t1,
    hA_ang_t1, hB_ang_t1,
    0.0, 0.0,
    0.0, penalty,
    1e30,
    -fric_limit, fric_limit,
    0.0,
  );

  // Friction tangent 2
  let jA_lin_t2 = t2;
  let jA_ang_t2 = cross3(rA, t2);
  let jB_lin_t2 = -t2;
  let jB_ang_t2 = -cross3(rB, t2);
  let hA_ang_t2 = -rA * dot(rA, t2) / max(dot(rA, rA), 1e-8);
  let hB_ang_t2 = -rB * dot(rB, t2) / max(dot(rB, rB), 1e-8);

  write_constraint_3d(
    row_base + 2u,
    bodyA, bodyB,
    jA_lin_t2, jA_ang_t2,
    jB_lin_t2, jB_ang_t2,
    hA_ang_t2, hB_ang_t2,
    0.0, 0.0,
    0.0, penalty,
    1e30,
    -fric_limit, fric_limit,
    0.0,
  );
}
