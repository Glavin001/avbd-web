// ─── 2D Constraint Assembly ──────────────────────────────────────────────────
// Converts contact points from narrow phase into solver constraint rows.
// Each contact generates 2 rows: normal (non-penetration) + friction (tangent).
//
// Contact input format (10 u32s per contact):
//   [bodyA, bodyB, featureId, _pad, normal_x, normal_y, depth, mu, cpx, cpy]
//
// Constraint output format (28 floats per row, matching existing primal/dual):
//   [bodyA(i32), bodyB(i32), forceType(u32), _pad(u32),
//    jacobian_a(vec4: dx,dy,dtheta,0), jacobian_b(vec4: dx,dy,dtheta,0),
//    hessian_diag_a(vec4: 0,0,angular,mu), hessian_diag_b(vec4: 0,0,angular,0),
//    c, c0, lambda, penalty, stiffness, fmin, fmax, active(u32)]

struct AssemblyParams {
  max_contacts: u32,
  max_constraints: u32,
  collision_margin: f32,
  dt: f32,
  penalty_min: f32,
  alpha: f32,
  warmstart_cache_size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: AssemblyParams;
@group(0) @binding(1) var<storage, read> contact_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> body_state: array<f32>;      // 8 floats/body
@group(0) @binding(3) var<storage, read> collider_info: array<u32>;   // 8 u32s/body
@group(0) @binding(4) var<storage, read_write> constraint_buffer: array<f32>;
@group(0) @binding(5) var<storage, read_write> constraint_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> warmstart_keys: array<u32>;   // 3 u32 per entry
@group(0) @binding(7) var<storage, read_write> warmstart_vals: array<f32>;   // 4 f32 per entry
@group(0) @binding(8) var<storage, read_write> warmstart_age: array<u32>;

const BODY_STRIDE: u32 = 8u;
const CONTACT_STRIDE: u32 = 10u;
const CONSTRAINT_STRIDE: u32 = 28u;
const FORCE_TYPE_CONTACT: u32 = 0u;

fn load_contact_f32(contact_idx: u32, field: u32) -> f32 {
  return bitcast<f32>(contact_buffer[contact_idx * CONTACT_STRIDE + field]);
}

fn load_contact_u32(contact_idx: u32, field: u32) -> u32 {
  return contact_buffer[contact_idx * CONTACT_STRIDE + field];
}

// Hash function for warmstart cache lookup
fn warmstart_hash(bodyA: u32, bodyB: u32, featureId: u32) -> u32 {
  var h = bodyA * 73856093u;
  h = h ^ (bodyB * 19349663u);
  h = h ^ (featureId * 83492791u);
  return h % params.warmstart_cache_size;
}

// Look up cached penalty value for this contact
fn warmstart_lookup(bodyA: u32, bodyB: u32, featureId: u32) -> f32 {
  if (params.warmstart_cache_size == 0u) {
    return params.penalty_min;
  }
  var slot = warmstart_hash(bodyA, bodyB, featureId);
  // Linear probing (max 8 attempts)
  for (var probe = 0u; probe < 8u; probe++) {
    let idx = slot * 3u;
    if (warmstart_keys[idx] == bodyA &&
        warmstart_keys[idx + 1u] == bodyB &&
        warmstart_keys[idx + 2u] == featureId) {
      let age = warmstart_age[slot];
      if (age < 5u) {
        // Return cached penalty, clamped
        return max(params.penalty_min, warmstart_vals[slot * 4u]);
      }
    }
    slot = (slot + 1u) % params.warmstart_cache_size;
  }
  return params.penalty_min;
}

fn write_constraint(row_idx: u32,
                    bodyA: i32, bodyB: i32,
                    jA: vec3<f32>, jB: vec3<f32>,
                    hA_ang: f32, hB_ang: f32,
                    c_val: f32, c0_val: f32,
                    lambda: f32, penalty: f32,
                    stiffness: f32,
                    fmin_val: f32, fmax_val: f32,
                    mu: f32) {
  let base = row_idx * CONSTRAINT_STRIDE;

  // Body indices + force type (as u32 bits stored in f32 slots)
  constraint_buffer[base + 0u] = bitcast<f32>(bodyA);
  constraint_buffer[base + 1u] = bitcast<f32>(bodyB);
  constraint_buffer[base + 2u] = bitcast<f32>(FORCE_TYPE_CONTACT);
  constraint_buffer[base + 3u] = 0.0;

  // Jacobian A (vec4)
  constraint_buffer[base + 4u] = jA.x;
  constraint_buffer[base + 5u] = jA.y;
  constraint_buffer[base + 6u] = jA.z;
  constraint_buffer[base + 7u] = 0.0;

  // Jacobian B (vec4)
  constraint_buffer[base + 8u] = jB.x;
  constraint_buffer[base + 9u] = jB.y;
  constraint_buffer[base + 10u] = jB.z;
  constraint_buffer[base + 11u] = 0.0;

  // Hessian diag A (vec4: 0, 0, angular, 0)
  constraint_buffer[base + 12u] = 0.0;
  constraint_buffer[base + 13u] = 0.0;
  constraint_buffer[base + 14u] = hA_ang;
  constraint_buffer[base + 15u] = 0.0;

  // Hessian diag B (vec4: 0, 0, angular, mu_for_friction_coupling)
  constraint_buffer[base + 16u] = 0.0;
  constraint_buffer[base + 17u] = 0.0;
  constraint_buffer[base + 18u] = hB_ang;
  constraint_buffer[base + 19u] = mu;

  // Scalar fields
  constraint_buffer[base + 20u] = c_val;
  constraint_buffer[base + 21u] = c0_val;
  constraint_buffer[base + 22u] = lambda;
  constraint_buffer[base + 23u] = penalty;
  constraint_buffer[base + 24u] = stiffness;
  constraint_buffer[base + 25u] = fmin_val;
  constraint_buffer[base + 26u] = fmax_val;
  constraint_buffer[base + 27u] = bitcast<f32>(1u); // active = true
}

@compute @workgroup_size(256)
fn constraint_assembly_2d(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let contact_idx = gid.x;
  if (contact_idx >= params.max_contacts) {
    return;
  }

  // Check if this contact is valid (bodyA == 0xFFFFFFFF sentinel for cleared slots)
  let bodyA_u32 = load_contact_u32(contact_idx, 0u);
  if (bodyA_u32 == 0xFFFFFFFFu) {
    return;
  }

  let bodyA = i32(bodyA_u32);
  let bodyB = i32(load_contact_u32(contact_idx, 1u));
  let featureId = load_contact_u32(contact_idx, 2u);

  let nx = load_contact_f32(contact_idx, 4u);
  let ny = load_contact_f32(contact_idx, 5u);
  let depth = load_contact_f32(contact_idx, 6u);
  let mu = load_contact_f32(contact_idx, 7u);

  // Load body positions
  let posA_x = body_state[u32(bodyA) * BODY_STRIDE + 0u];
  let posA_y = body_state[u32(bodyA) * BODY_STRIDE + 1u];
  let posB_x = body_state[u32(bodyB) * BODY_STRIDE + 0u];
  let posB_y = body_state[u32(bodyB) * BODY_STRIDE + 1u];

  // Read actual contact position from narrowphase output
  let cpx = load_contact_f32(contact_idx, 8u);
  let cpy = load_contact_f32(contact_idx, 9u);

  // Lever arms
  let rA_x = cpx - posA_x;
  let rA_y = cpy - posA_y;
  let rB_x = cpx - posB_x;
  let rB_y = cpy - posB_y;

  // Normal constraint Jacobians
  // J_A = [n.x, n.y, rA × n]
  let torqueA_n = rA_x * ny - rA_y * nx;
  // J_B = [-n.x, -n.y, -(rB × n)]
  let torqueB_n = -(rB_x * ny - rB_y * nx);

  let jA_normal = vec3<f32>(nx, ny, torqueA_n);
  let jB_normal = vec3<f32>(-nx, -ny, torqueB_n);

  // Geometric stiffness (Hessian diagonal angular term)
  let hessA_ang_n = -(rA_x * nx + rA_y * ny);
  let hessB_ang_n = -(rB_x * nx + rB_y * ny);

  // Constraint value: C = -depth + margin
  let c_normal = -depth + params.collision_margin;
  let c0_normal = c_normal;

  // Warmstart lookup
  let penalty = warmstart_lookup(bodyA_u32, u32(bodyB), featureId);

  // Allocate 2 constraint rows atomically
  let row_base = atomicAdd(&constraint_count[0], 2u);
  if (row_base + 2u > params.max_constraints) {
    return; // Buffer full
  }

  // Write normal row
  write_constraint(
    row_base,
    bodyA, bodyB,
    jA_normal, jB_normal,
    hessA_ang_n, hessB_ang_n,
    c_normal, c0_normal,
    0.0, penalty,
    1e30,        // stiffness (effectively infinite for contacts)
    -1e30, 0.0,  // fmin, fmax (compressive only)
    mu,
  );

  // Friction row
  let tx = -ny;  // tangent perpendicular to normal
  let ty = nx;

  let torqueA_t = rA_x * ty - rA_y * tx;
  let torqueB_t = -(rB_x * ty - rB_y * tx);

  let jA_friction = vec3<f32>(tx, ty, torqueA_t);
  let jB_friction = vec3<f32>(-tx, -ty, torqueB_t);

  let hessA_ang_t = -(rA_x * tx + rA_y * ty);
  let hessB_ang_t = -(rB_x * tx + rB_y * ty);

  // Estimated friction bounds
  let est_normal_force = penalty * depth;
  let fric_limit = mu * est_normal_force;

  write_constraint(
    row_base + 1u,
    bodyA, bodyB,
    jA_friction, jB_friction,
    hessA_ang_t, hessB_ang_t,
    0.0, 0.0,
    0.0, penalty,
    1e30,
    -fric_limit, fric_limit,
    0.0, // mu stored only in normal row
  );
}
