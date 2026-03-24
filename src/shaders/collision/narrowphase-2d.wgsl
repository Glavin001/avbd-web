// ─── AVBD 2D Narrow Phase Compute Shader ─────────────────────────────────────
// One thread per candidate collision pair. Determines shape types, runs the
// appropriate collision algorithm (box-box SAT, box-circle, circle-circle),
// and writes contact points to the output buffer via atomic append.

struct NarrowphaseParams2D {
  num_pairs: u32,
  max_contacts: u32,
  collision_margin: f32,
  _pad: u32,
}

// Shape types
const SHAPE_CUBOID: u32 = 0u;
const SHAPE_BALL: u32 = 1u;

const EPSILON: f32 = 0.0001;
const COLLISION_MARGIN: f32 = 0.0005;

@group(0) @binding(0) var<uniform> params: NarrowphaseParams2D;
@group(0) @binding(1) var<storage, read> pair_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> body_state: array<f32>;
@group(0) @binding(3) var<storage, read> collider_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> contact_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> contact_count: array<atomic<u32>>;

// ─── Body state accessors (8 floats per body) ───────────────────────────────

fn body_pos(idx: u32) -> vec2<f32> {
  let base = idx * 8u;
  return vec2<f32>(body_state[base], body_state[base + 1u]);
}

fn body_angle(idx: u32) -> f32 {
  return body_state[idx * 8u + 2u];
}

// ─── Collider info accessors (8 u32s per body) ─────────────────────────────

fn collider_shape(idx: u32) -> u32 {
  return collider_info[idx * 8u];
}

fn collider_half_ext(idx: u32) -> vec2<f32> {
  let base = idx * 8u;
  return vec2<f32>(
    bitcast<f32>(collider_info[base + 1u]),
    bitcast<f32>(collider_info[base + 2u]),
  );
}

fn collider_radius(idx: u32) -> f32 {
  return bitcast<f32>(collider_info[idx * 8u + 3u]);
}

fn collider_friction(idx: u32) -> f32 {
  return bitcast<f32>(collider_info[idx * 8u + 4u]);
}

// ─── Contact output helpers ─────────────────────────────────────────────────
// Contact format: 8 u32s = [bodyA, bodyB, featureId, _pad, nx, ny, depth, mu]

fn emit_contact(
  body_a: u32, body_b: u32, feature_id: u32,
  normal: vec2<f32>, depth: f32, mu: f32,
) {
  let slot = atomicAdd(&contact_count[0], 1u);
  if (slot >= params.max_contacts) {
    return;
  }
  let base = slot * 8u;
  contact_buffer[base + 0u] = body_a;
  contact_buffer[base + 1u] = body_b;
  contact_buffer[base + 2u] = feature_id;
  contact_buffer[base + 3u] = 0u;
  contact_buffer[base + 4u] = bitcast<u32>(normal.x);
  contact_buffer[base + 5u] = bitcast<u32>(normal.y);
  contact_buffer[base + 6u] = bitcast<u32>(depth);
  contact_buffer[base + 7u] = bitcast<u32>(mu);
}

// ─── 2D rotation helpers ────────────────────────────────────────────────────

fn rotate2d(v: vec2<f32>, cos_a: f32, sin_a: f32) -> vec2<f32> {
  return vec2<f32>(cos_a * v.x - sin_a * v.y, sin_a * v.x + cos_a * v.y);
}

fn inv_rotate2d(v: vec2<f32>, cos_a: f32, sin_a: f32) -> vec2<f32> {
  return vec2<f32>(cos_a * v.x + sin_a * v.y, -sin_a * v.x + cos_a * v.y);
}

// ─── Box vertices in world space ────────────────────────────────────────────

fn get_box_vertex(pos: vec2<f32>, he: vec2<f32>, c: f32, s: f32, corner: u32) -> vec2<f32> {
  var lx: f32;
  var ly: f32;
  switch (corner) {
    case 0u: { lx = -he.x; ly = -he.y; }
    case 1u: { lx =  he.x; ly = -he.y; }
    case 2u: { lx =  he.x; ly =  he.y; }
    default: { lx = -he.x; ly =  he.y; }
  }
  return pos + vec2<f32>(c * lx - s * ly, s * lx + c * ly);
}

// ─── Project polygon onto axis ──────────────────────────────────────────────

fn project_box(pos: vec2<f32>, he: vec2<f32>, c: f32, s: f32, axis: vec2<f32>) -> vec2<f32> {
  // Returns vec2(min, max) projection
  var mn = 1e30f;
  var mx = -1e30f;
  for (var i = 0u; i < 4u; i++) {
    let v = get_box_vertex(pos, he, c, s, i);
    let d = dot(axis, v);
    mn = min(mn, d);
    mx = max(mx, d);
  }
  return vec2<f32>(mn, mx);
}

// ─── Support edge: find the edge most anti-aligned with direction ───────────

struct Edge2D {
  p0: vec2<f32>,
  p1: vec2<f32>,
  idx0: u32,
  idx1: u32,
}

fn find_support_edge(
  pos: vec2<f32>, he: vec2<f32>, c: f32, s: f32, direction: vec2<f32>,
) -> Edge2D {
  // Find vertex most aligned with direction
  var best_dot = -1e30f;
  var best_idx = 0u;
  var verts: array<vec2<f32>, 4>;
  for (var i = 0u; i < 4u; i++) {
    verts[i] = get_box_vertex(pos, he, c, s, i);
    let d = dot(verts[i], direction);
    if (d > best_dot) {
      best_dot = d;
      best_idx = i;
    }
  }

  let prev_idx = (best_idx + 3u) % 4u;
  let next_idx = (best_idx + 1u) % 4u;

  let edge_prev = normalize(verts[best_idx] - verts[prev_idx]);
  let edge_next = normalize(verts[next_idx] - verts[best_idx]);

  var result: Edge2D;
  if (abs(dot(edge_prev, direction)) < abs(dot(edge_next, direction))) {
    result.p0 = verts[prev_idx];
    result.p1 = verts[best_idx];
    result.idx0 = prev_idx;
    result.idx1 = best_idx;
  } else {
    result.p0 = verts[best_idx];
    result.p1 = verts[next_idx];
    result.idx0 = best_idx;
    result.idx1 = next_idx;
  }
  return result;
}

// ─── Clip segment against half-plane: dot(planeNormal, x) + planeOffset <= 0

struct ClipResult {
  p: array<vec2<f32>, 2>,
  count: u32,
}

fn clip_edge(
  p0: vec2<f32>, p1: vec2<f32>,
  plane_normal: vec2<f32>, plane_offset: f32,
) -> ClipResult {
  let d0 = dot(plane_normal, p0) + plane_offset;
  let d1 = dot(plane_normal, p1) + plane_offset;

  var result: ClipResult;
  result.count = 0u;

  if (d0 <= 0.0) {
    result.p[result.count] = p0;
    result.count += 1u;
  }
  if (d1 <= 0.0) {
    result.p[result.count] = p1;
    result.count += 1u;
  }
  if (d0 * d1 < 0.0) {
    let t = d0 / (d0 - d1);
    result.p[result.count] = p0 + t * (p1 - p0);
    result.count += 1u;
  }

  return result;
}

// ─── Box-Box 2D SAT ────────────────────────────────────────────────────────

fn collide_box_box(
  idx_a: u32, idx_b: u32, mu: f32,
) {
  let pos_a = body_pos(idx_a);
  let pos_b = body_pos(idx_b);
  let he_a = collider_half_ext(idx_a);
  let he_b = collider_half_ext(idx_b);
  let angle_a = body_angle(idx_a);
  let angle_b = body_angle(idx_b);

  let ca = cos(angle_a);
  let sa = sin(angle_a);
  let cb = cos(angle_b);
  let sb = sin(angle_b);

  // 4 separating axes: 2 from each box
  var axes: array<vec2<f32>, 4>;
  axes[0] = vec2<f32>(ca, sa);
  axes[1] = vec2<f32>(-sa, ca);
  axes[2] = vec2<f32>(cb, sb);
  axes[3] = vec2<f32>(-sb, cb);

  var min_overlap = 1e30f;
  var best_axis = vec2<f32>(0.0, 0.0);
  var best_axis_idx = 0u;

  for (var i = 0u; i < 4u; i++) {
    let axis = axes[i];
    let proj_a = project_box(pos_a, he_a, ca, sa, axis);
    let proj_b = project_box(pos_b, he_b, cb, sb, axis);

    let overlap = min(proj_a.y - proj_b.x, proj_b.y - proj_a.x);
    if (overlap < 0.0) {
      return; // Separating axis found
    }

    if (overlap < min_overlap) {
      min_overlap = overlap;
      best_axis = axis;
      best_axis_idx = i;
    }
  }

  // Ensure normal points from B to A
  let d = pos_a - pos_b;
  if (dot(d, best_axis) < 0.0) {
    best_axis = -best_axis;
  }

  // Find support edges
  let edge_a = find_support_edge(pos_a, he_a, ca, sa, -best_axis);
  let edge_b = find_support_edge(pos_b, he_b, cb, sb, best_axis);

  // Determine reference and incident edges
  let edge_a_dir = normalize(edge_a.p1 - edge_a.p0);
  let edge_b_dir = normalize(edge_b.p1 - edge_b.p0);
  let dot_a = abs(dot(edge_a_dir, best_axis));
  let dot_b = abs(dot(edge_b_dir, best_axis));

  var ref_p0: vec2<f32>;
  var ref_p1: vec2<f32>;
  var inc_p0: vec2<f32>;
  var inc_p1: vec2<f32>;
  var ref_normal: vec2<f32>;
  var flip = false;
  var ref_box: u32;

  if (dot_a <= dot_b) {
    // Edge A is reference
    ref_p0 = edge_a.p0;
    ref_p1 = edge_a.p1;
    inc_p0 = edge_b.p0;
    inc_p1 = edge_b.p1;
    ref_normal = best_axis;
    flip = false;
    ref_box = 0u;
  } else {
    // Edge B is reference
    ref_p0 = edge_b.p0;
    ref_p1 = edge_b.p1;
    inc_p0 = edge_a.p0;
    inc_p1 = edge_a.p1;
    ref_normal = -best_axis;
    flip = true;
    ref_box = 1u;
  }

  // Clip incident edge against reference edge's side planes
  let ref_dir = normalize(ref_p1 - ref_p0);

  // Clip against left side: -ref_dir
  let neg_ref_dir = -ref_dir;
  let left_offset = -dot(neg_ref_dir, ref_p0);
  var clipped = clip_edge(inc_p0, inc_p1, neg_ref_dir, left_offset);
  if (clipped.count < 2u) {
    // Fallback: midpoint contact
    let mid = (pos_a + pos_b) * 0.5;
    let feature_id = (ref_box << 4u) | (best_axis_idx << 2u);
    emit_contact(idx_a, idx_b, feature_id, best_axis, min_overlap + COLLISION_MARGIN, mu);
    return;
  }

  // Clip against right side: ref_dir
  let right_offset = -dot(ref_dir, ref_p1);
  clipped = clip_edge(clipped.p[0], clipped.p[1], ref_dir, right_offset);
  if (clipped.count < 2u) {
    let mid = (pos_a + pos_b) * 0.5;
    let feature_id = (ref_box << 4u) | (best_axis_idx << 2u);
    emit_contact(idx_a, idx_b, feature_id, best_axis, min_overlap + COLLISION_MARGIN, mu);
    return;
  }

  // Keep only points behind the reference face
  let face_normal = select(ref_normal, -ref_normal, flip);
  let ref_face_offset = dot(face_normal, ref_p0);

  var emitted = 0u;
  for (var i = 0u; i < clipped.count; i++) {
    let p = clipped.p[i];
    let sep = dot(face_normal, p) - ref_face_offset;
    if (sep <= min_overlap + 0.01) {
      var contact_depth: f32;
      if (sep < -1e-6) {
        contact_depth = -sep;
      } else {
        contact_depth = min_overlap;
      }
      let contact_normal = select(best_axis, -best_axis, flip);
      // Feature ID: (ref_box << 4) | (axis_index << 2) | clip_vertex_index
      let feature_id = (ref_box << 4u) | (best_axis_idx << 2u) | i;
      emit_contact(idx_a, idx_b, feature_id, best_axis, contact_depth + COLLISION_MARGIN, mu);
      emitted += 1u;
    }
  }

  // Fallback if no contacts were emitted
  if (emitted == 0u) {
    let mid = (pos_a + pos_b) * 0.5;
    let feature_id = (ref_box << 4u) | (best_axis_idx << 2u);
    emit_contact(idx_a, idx_b, feature_id, best_axis, min_overlap + COLLISION_MARGIN, mu);
  }
}

// ─── Box-Circle 2D ──────────────────────────────────────────────────────────

fn collide_box_circle(
  box_idx: u32, circle_idx: u32, mu: f32, swap: bool,
) {
  let box_pos = body_pos(box_idx);
  let circle_pos = body_pos(circle_idx);
  let he = collider_half_ext(box_idx);
  let radius = collider_radius(circle_idx);
  let angle = body_angle(box_idx);

  let ca = cos(angle);
  let sa = sin(angle);

  // Transform circle center to box local space
  let d = circle_pos - box_pos;
  let local_center = inv_rotate2d(d, ca, sa);

  // Clamp to box half-extents
  let closest = clamp(local_center, -he, he);
  let diff = local_center - closest;
  let dist_sq = dot(diff, diff);

  if (dist_sq >= radius * radius) {
    return;
  }

  let dist = sqrt(dist_sq);
  var local_normal: vec2<f32>;
  var contact_depth: f32;

  if (dist > 1e-10) {
    local_normal = diff / dist;
    contact_depth = radius - dist;
  } else {
    // Circle center is inside the box
    let dx = he.x - abs(local_center.x);
    let dy = he.y - abs(local_center.y);
    if (dx < dy) {
      local_normal = select(vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0), local_center.x >= 0.0);
      contact_depth = dx + radius;
    } else {
      local_normal = select(vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0), local_center.y >= 0.0);
      contact_depth = dy + radius;
    }
  }

  // Transform normal back to world space (local_normal points from box surface to circle)
  let outward = rotate2d(local_normal, ca, sa);
  // Convention: normal from B to A
  // If not swapped: box=A, circle=B, normal = -(outward) = from circle to box
  // If swapped: circle=A, box=B, normal = outward = from box to circle
  var manifold_normal: vec2<f32>;
  var out_a: u32;
  var out_b: u32;

  if (swap) {
    manifold_normal = outward;
    out_a = circle_idx;
    out_b = box_idx;
  } else {
    manifold_normal = -outward;
    out_a = box_idx;
    out_b = circle_idx;
  }

  let feature_id = 0x100u; // box-circle marker
  emit_contact(out_a, out_b, feature_id, manifold_normal, contact_depth + COLLISION_MARGIN, mu);
}

// ─── Circle-Circle 2D ───────────────────────────────────────────────────────

fn collide_circle_circle(
  idx_a: u32, idx_b: u32, mu: f32,
) {
  let pos_a = body_pos(idx_a);
  let pos_b = body_pos(idx_b);
  let radius_a = collider_radius(idx_a);
  let radius_b = collider_radius(idx_b);

  let d = pos_a - pos_b;
  let dist = length(d);
  let combined = radius_a + radius_b;

  if (dist >= combined) {
    return;
  }

  var normal: vec2<f32>;
  if (dist > 1e-10) {
    normal = d / dist;
  } else {
    normal = vec2<f32>(0.0, 1.0);
  }

  let depth = combined - dist + COLLISION_MARGIN;
  let feature_id = 0x200u; // circle-circle marker
  emit_contact(idx_a, idx_b, feature_id, normal, depth, mu);
}

// ─── Main entry point ───────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn narrowphase_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_idx = gid.x;
  if (pair_idx >= params.num_pairs) {
    return;
  }

  let body_a = pair_buffer[pair_idx * 2u];
  let body_b = pair_buffer[pair_idx * 2u + 1u];
  let shape_a = collider_shape(body_a);
  let shape_b = collider_shape(body_b);

  let friction_a = collider_friction(body_a);
  let friction_b = collider_friction(body_b);
  let mu = sqrt(friction_a * friction_b);

  if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_CUBOID) {
    collide_box_box(body_a, body_b, mu);
  } else if (shape_a == SHAPE_BALL && shape_b == SHAPE_BALL) {
    collide_circle_circle(body_a, body_b, mu);
  } else if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_BALL) {
    collide_box_circle(body_a, body_b, mu, false);
  } else if (shape_a == SHAPE_BALL && shape_b == SHAPE_CUBOID) {
    collide_box_circle(body_b, body_a, mu, true);
  }
}
