// Morton code generation for 2D LBVH construction.
// Computes per-body AABBs and 32-bit Morton codes from centroid positions.

struct Params {
  num_bodies:   u32,
  scene_min_x:  f32,
  scene_min_y:  f32,
  scene_max_x:  f32,
  scene_max_y:  f32,
  _pad0:        u32,
  _pad1:        u32,
  _pad2:        u32,
};

// Body state: 8 floats per body [x, y, angle, vx, vy, omega, mass, inertia]
@group(0) @binding(0) var<storage, read> body_state: array<f32>;

// Collider info: 8 u32s per body [shape_type, halfExtX, halfExtY, halfExtZ, radius, friction, restitution, bodyType]
@group(0) @binding(1) var<storage, read> collider_info: array<u32>;

@group(0) @binding(2) var<uniform> params: Params;

// Output: one u32 Morton code per body
@group(0) @binding(3) var<storage, read_write> morton_codes: array<u32>;

// Output: 4 floats per body [minX, minY, maxX, maxY]
@group(0) @binding(4) var<storage, read_write> aabb_buffer: array<f32>;

// Spread 16 bits with 1-bit gaps: 0b...abcd -> 0b...0a0b0c0d
fn expandBits2D(v: u32) -> u32 {
  var x = v & 0x0000FFFFu;
  x = (x | (x << 8u)) & 0x00FF00FFu;
  x = (x | (x << 4u)) & 0x0F0F0F0Fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}

fn morton2D(nx: f32, ny: f32) -> u32 {
  // Quantize normalised [0,1] coordinates to 16-bit integers
  let ix = min(u32(nx * 65535.0), 65535u);
  let iy = min(u32(ny * 65535.0), 65535u);
  return expandBits2D(ix) | (expandBits2D(iy) << 1u);
}

@compute @workgroup_size(256)
fn morton_codes_2d(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.num_bodies) {
    return;
  }

  // Read body position and rotation
  let base = idx * 8u;
  let px    = body_state[base + 0u];
  let py    = body_state[base + 1u];
  let angle = body_state[base + 2u];

  // Read collider info (8 u32s per body, matching COLLIDER_INFO_STRIDE)
  let ci_base    = idx * 8u;
  let shape_type = collider_info[ci_base + 0u];
  let half_ext_x = bitcast<f32>(collider_info[ci_base + 1u]);
  let half_ext_y = bitcast<f32>(collider_info[ci_base + 2u]);
  let radius     = bitcast<f32>(collider_info[ci_base + 4u]);

  // Compute AABB
  var aabb_min_x: f32;
  var aabb_min_y: f32;
  var aabb_max_x: f32;
  var aabb_max_y: f32;

  if (shape_type == 0u) {
    // Cuboid (box): compute rotated extents
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    // Half-extents of AABB = |R| * half_extents  (absolute value of rotation matrix columns)
    let ex = abs(cos_a) * half_ext_x + abs(sin_a) * half_ext_y;
    let ey = abs(sin_a) * half_ext_x + abs(cos_a) * half_ext_y;
    aabb_min_x = px - ex;
    aabb_min_y = py - ey;
    aabb_max_x = px + ex;
    aabb_max_y = py + ey;
  } else {
    // Ball (circle): AABB is position ± radius
    aabb_min_x = px - radius;
    aabb_min_y = py - radius;
    aabb_max_x = px + radius;
    aabb_max_y = py + radius;
  }

  // Write AABB
  let aabb_base = idx * 4u;
  aabb_buffer[aabb_base + 0u] = aabb_min_x;
  aabb_buffer[aabb_base + 1u] = aabb_min_y;
  aabb_buffer[aabb_base + 2u] = aabb_max_x;
  aabb_buffer[aabb_base + 3u] = aabb_max_y;

  // Compute centroid
  let cx = (aabb_min_x + aabb_max_x) * 0.5;
  let cy = (aabb_min_y + aabb_max_y) * 0.5;

  // Normalise centroid to [0,1]^2 within scene bounds
  let inv_x = select(0.0, 1.0 / (params.scene_max_x - params.scene_min_x),
                     params.scene_max_x > params.scene_min_x);
  let inv_y = select(0.0, 1.0 / (params.scene_max_y - params.scene_min_y),
                     params.scene_max_y > params.scene_min_y);

  let nx = clamp((cx - params.scene_min_x) * inv_x, 0.0, 1.0);
  let ny = clamp((cy - params.scene_min_y) * inv_y, 0.0, 1.0);

  morton_codes[idx] = morton2D(nx, ny);
}
