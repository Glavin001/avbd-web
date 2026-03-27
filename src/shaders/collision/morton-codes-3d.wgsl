// Morton code generation for 3D LBVH construction.
// Computes per-body AABBs and 32-bit Morton codes (10+10+10 bits) from centroid positions.

struct Params {
  num_bodies:  u32,
  scene_min_x: f32,
  scene_min_y: f32,
  scene_min_z: f32,
  scene_max_x: f32,
  scene_max_y: f32,
  scene_max_z: f32,
  _pad0:       u32,
};

// Body state: 20 floats per body
// [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, mass, Ix, Iy, Iz, pad, pad, pad]
@group(0) @binding(0) var<storage, read> body_state: array<f32>;

// Collider info: 8 values per body
// [shape_type(u32), halfExtentX, halfExtentY, halfExtentZ, radius, _pad, _pad, _pad]
@group(0) @binding(1) var<storage, read> collider_info: array<u32>;

@group(0) @binding(2) var<uniform> params: Params;

// Output: one u32 Morton code per body
@group(0) @binding(3) var<storage, read_write> morton_codes: array<u32>;

// Output: 6 floats per body [minX, minY, minZ, maxX, maxY, maxZ]
@group(0) @binding(4) var<storage, read_write> aabb_buffer: array<f32>;

// Spread 10 bits with 2-bit gaps for 3D interleaving
fn expandBits3D(v: u32) -> u32 {
  var x = v & 0x000003FFu;  // mask to 10 bits
  x = (x * 0x00010001u) & 0xFF0000FFu;
  x = (x * 0x00000101u) & 0x0F00F00Fu;
  x = (x * 0x00000011u) & 0xC30C30C3u;
  x = (x * 0x00000005u) & 0x49249249u;
  return x;
}

fn morton3D(nx: f32, ny: f32, nz: f32) -> u32 {
  // Quantize normalised [0,1] coordinates to 10-bit integers
  let ix = min(u32(nx * 1023.0), 1023u);
  let iy = min(u32(ny * 1023.0), 1023u);
  let iz = min(u32(nz * 1023.0), 1023u);
  return expandBits3D(ix) | (expandBits3D(iy) << 1u) | (expandBits3D(iz) << 2u);
}

@compute @workgroup_size(256)
fn morton_codes_3d(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.num_bodies) {
    return;
  }

  // Read body position and quaternion
  let base = idx * 20u;
  let px = body_state[base + 0u];
  let py = body_state[base + 1u];
  let pz = body_state[base + 2u];
  let qw = body_state[base + 3u];
  let qx = body_state[base + 4u];
  let qy = body_state[base + 5u];
  let qz = body_state[base + 6u];

  // Read collider info
  let ci_base    = idx * 8u;
  let shape_type = collider_info[ci_base + 0u];
  let half_ext_x = bitcast<f32>(collider_info[ci_base + 1u]);
  let half_ext_y = bitcast<f32>(collider_info[ci_base + 2u]);
  let half_ext_z = bitcast<f32>(collider_info[ci_base + 3u]);
  let radius     = bitcast<f32>(collider_info[ci_base + 4u]);

  // Compute AABB
  var aabb_min_x: f32;
  var aabb_min_y: f32;
  var aabb_min_z: f32;
  var aabb_max_x: f32;
  var aabb_max_y: f32;
  var aabb_max_z: f32;

  if (shape_type == 0u) {
    // Cuboid: compute rotation matrix from quaternion, then rotated extents
    // R = I + 2*qw*[q]x + 2*[q]x*[q]x  (via standard formula)
    let xx = qx * qx;
    let yy = qy * qy;
    let zz = qz * qz;
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    // Rotation matrix columns (row-major entries)
    let r00 = 1.0 - 2.0 * (yy + zz);
    let r01 = 2.0 * (xy - wz);
    let r02 = 2.0 * (xz + wy);
    let r10 = 2.0 * (xy + wz);
    let r11 = 1.0 - 2.0 * (xx + zz);
    let r12 = 2.0 * (yz - wx);
    let r20 = 2.0 * (xz - wy);
    let r21 = 2.0 * (yz + wx);
    let r22 = 1.0 - 2.0 * (xx + yy);

    // AABB half-extents = |R| * h  (absolute rotation matrix times half-extents)
    let ex = abs(r00) * half_ext_x + abs(r01) * half_ext_y + abs(r02) * half_ext_z;
    let ey = abs(r10) * half_ext_x + abs(r11) * half_ext_y + abs(r12) * half_ext_z;
    let ez = abs(r20) * half_ext_x + abs(r21) * half_ext_y + abs(r22) * half_ext_z;

    aabb_min_x = px - ex;
    aabb_min_y = py - ey;
    aabb_min_z = pz - ez;
    aabb_max_x = px + ex;
    aabb_max_y = py + ey;
    aabb_max_z = pz + ez;
  } else {
    // Sphere: AABB is position ± radius
    aabb_min_x = px - radius;
    aabb_min_y = py - radius;
    aabb_min_z = pz - radius;
    aabb_max_x = px + radius;
    aabb_max_y = py + radius;
    aabb_max_z = pz + radius;
  }

  // Write AABB
  let aabb_base = idx * 6u;
  aabb_buffer[aabb_base + 0u] = aabb_min_x;
  aabb_buffer[aabb_base + 1u] = aabb_min_y;
  aabb_buffer[aabb_base + 2u] = aabb_min_z;
  aabb_buffer[aabb_base + 3u] = aabb_max_x;
  aabb_buffer[aabb_base + 4u] = aabb_max_y;
  aabb_buffer[aabb_base + 5u] = aabb_max_z;

  // Compute centroid
  let cx = (aabb_min_x + aabb_max_x) * 0.5;
  let cy = (aabb_min_y + aabb_max_y) * 0.5;
  let cz = (aabb_min_z + aabb_max_z) * 0.5;

  // Normalise centroid to [0,1]^3 within scene bounds
  let inv_x = select(0.0, 1.0 / (params.scene_max_x - params.scene_min_x),
                     params.scene_max_x > params.scene_min_x);
  let inv_y = select(0.0, 1.0 / (params.scene_max_y - params.scene_min_y),
                     params.scene_max_y > params.scene_min_y);
  let inv_z = select(0.0, 1.0 / (params.scene_max_z - params.scene_min_z),
                     params.scene_max_z > params.scene_min_z);

  let nx = clamp((cx - params.scene_min_x) * inv_x, 0.0, 1.0);
  let ny = clamp((cy - params.scene_min_y) * inv_y, 0.0, 1.0);
  let nz = clamp((cz - params.scene_min_z) * inv_z, 0.0, 1.0);

  morton_codes[idx] = morton3D(nx, ny, nz);
}
