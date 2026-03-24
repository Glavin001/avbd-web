// ─── AVBD 3D Narrow Phase Compute Shader ─────────────────────────────────────
// One thread per candidate collision pair. Determines shape types, runs the
// appropriate collision algorithm (box-box SAT + clipping, box-sphere,
// sphere-sphere), and writes contact points to the output buffer via atomic
// append.

struct NarrowphaseParams3D {
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
const MAX_CLIP_VERTS: u32 = 8u;

@group(0) @binding(0) var<uniform> params: NarrowphaseParams3D;
@group(0) @binding(1) var<storage, read> pair_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> body_state: array<f32>;
@group(0) @binding(3) var<storage, read> collider_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> contact_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> contact_count: array<atomic<u32>>;

// ─── Body state accessors (20 floats per body) ─────────────────────────────
// Layout: [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz, mass, Ix,Iy,Iz, ...]

fn body_pos3(idx: u32) -> vec3<f32> {
  let base = idx * 20u;
  return vec3<f32>(body_state[base], body_state[base + 1u], body_state[base + 2u]);
}

fn body_quat(idx: u32) -> vec4<f32> {
  let base = idx * 20u;
  // vec4(w, x, y, z)
  return vec4<f32>(
    body_state[base + 3u],
    body_state[base + 4u],
    body_state[base + 5u],
    body_state[base + 6u],
  );
}

// ─── Collider info accessors (8 u32s per body) ─────────────────────────────
// Layout: [shape(u32), heX(f32), heY(f32), heZ(f32), radius(f32), friction(f32), restitution(f32), bodyType(u32)]

fn collider_shape(idx: u32) -> u32 {
  return collider_info[idx * 8u];
}

fn collider_half_ext3(idx: u32) -> vec3<f32> {
  let base = idx * 8u;
  return vec3<f32>(
    bitcast<f32>(collider_info[base + 1u]),
    bitcast<f32>(collider_info[base + 2u]),
    bitcast<f32>(collider_info[base + 3u]),
  );
}

fn collider_radius3(idx: u32) -> f32 {
  return bitcast<f32>(collider_info[idx * 8u + 4u]);
}

fn collider_friction3(idx: u32) -> f32 {
  return bitcast<f32>(collider_info[idx * 8u + 5u]);
}

// ─── Contact output (12 u32s per contact) ───────────────────────────────────
// [bodyA, bodyB, featureId, _pad, nx, ny, nz, depth, px, py, pz, mu]

fn emit_contact3(
  body_a: u32, body_b: u32, feature_id: u32,
  normal: vec3<f32>, depth: f32,
  point: vec3<f32>, mu: f32,
) {
  let slot = atomicAdd(&contact_count[0], 1u);
  if (slot >= params.max_contacts) {
    return;
  }
  let base = slot * 12u;
  contact_buffer[base + 0u] = body_a;
  contact_buffer[base + 1u] = body_b;
  contact_buffer[base + 2u] = feature_id;
  contact_buffer[base + 3u] = 0u;
  contact_buffer[base + 4u] = bitcast<u32>(normal.x);
  contact_buffer[base + 5u] = bitcast<u32>(normal.y);
  contact_buffer[base + 6u] = bitcast<u32>(normal.z);
  contact_buffer[base + 7u] = bitcast<u32>(depth);
  contact_buffer[base + 8u] = bitcast<u32>(point.x);
  contact_buffer[base + 9u] = bitcast<u32>(point.y);
  contact_buffer[base + 10u] = bitcast<u32>(point.z);
  contact_buffer[base + 11u] = bitcast<u32>(mu);
}

// ─── Quaternion helpers ─────────────────────────────────────────────────────

// Rotate a vector by a quaternion q = (w, x, y, z)
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let w = q.x; // w component
  let u = vec3<f32>(q.y, q.z, q.w); // (x, y, z)
  let t = 2.0 * cross(u, v);
  return v + w * t + cross(u, t);
}

// Rotate by conjugate (inverse rotation)
fn quat_rotate_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let conj = vec4<f32>(q.x, -q.y, -q.z, -q.w);
  return quat_rotate(conj, v);
}

// Build rotation matrix columns from quaternion q = (w, x, y, z)
fn quat_to_axes(q: vec4<f32>) -> mat3x3<f32> {
  let w = q.x;
  let x = q.y;
  let y = q.z;
  let z = q.w;

  let x2 = x + x;
  let y2 = y + y;
  let z2 = z + z;
  let xx = x * x2;
  let xy = x * y2;
  let xz = x * z2;
  let yy = y * y2;
  let yz = y * z2;
  let zz = z * z2;
  let wx = w * x2;
  let wy = w * y2;
  let wz = w * z2;

  return mat3x3<f32>(
    vec3<f32>(1.0 - yy - zz, xy + wz, xz - wy),
    vec3<f32>(xy - wz, 1.0 - xx - zz, yz + wx),
    vec3<f32>(xz + wy, yz - wx, 1.0 - xx - yy),
  );
}

// ─── Polygon clipping buffer ────────────────────────────────────────────────

struct ClipPolygon {
  verts: array<vec3<f32>, 8>,
  count: u32,
}

fn clip_polygon_by_plane(poly: ClipPolygon, plane_n: vec3<f32>, plane_d: f32) -> ClipPolygon {
  var result: ClipPolygon;
  result.count = 0u;

  if (poly.count == 0u) {
    return result;
  }

  for (var i = 0u; i < poly.count; i++) {
    let a = poly.verts[i];
    let b = poly.verts[(i + 1u) % poly.count];
    let da = dot(a, plane_n) - plane_d;
    let db = dot(b, plane_n) - plane_d;

    if (da <= 0.0) {
      if (result.count < MAX_CLIP_VERTS) {
        result.verts[result.count] = a;
        result.count += 1u;
      }
    }

    // Check if edge crosses the plane (different signs)
    if ((da > 0.0) != (db > 0.0)) {
      let t = da / (da - db);
      let intersection = a + t * (b - a);
      if (result.count < MAX_CLIP_VERTS) {
        result.verts[result.count] = intersection;
        result.count += 1u;
      }
    }
  }

  return result;
}

// ─── Get 4 vertices of a box face in world space ────────────────────────────
// faceAxis: 0=x, 1=y, 2=z; sign: +1 or -1

fn get_box_face_verts(
  pos: vec3<f32>, axes: mat3x3<f32>, he: vec3<f32>,
  face_axis: u32, sign_val: f32,
) -> ClipPolygon {
  var result: ClipPolygon;
  result.count = 4u;

  let a1 = (face_axis + 1u) % 3u;
  let a2 = (face_axis + 2u) % 3u;

  let face_ax = vec3<f32>(axes[face_axis].x, axes[face_axis].y, axes[face_axis].z);
  let tang1 = vec3<f32>(axes[a1].x, axes[a1].y, axes[a1].z);
  let tang2 = vec3<f32>(axes[a2].x, axes[a2].y, axes[a2].z);

  let he_arr = array<f32, 3>(he.x, he.y, he.z);
  let center = pos + face_ax * sign_val * he_arr[face_axis];
  let t1 = tang1 * he_arr[a1];
  let t2 = tang2 * he_arr[a2];

  result.verts[0] = center + t1 + t2;
  result.verts[1] = center - t1 + t2;
  result.verts[2] = center - t1 - t2;
  result.verts[3] = center + t1 - t2;

  return result;
}

// ─── Find axis most aligned with a direction ────────────────────────────────

fn find_most_aligned_axis(axes: mat3x3<f32>, dir: vec3<f32>) -> u32 {
  var max_abs = -1.0f;
  var best = 0u;
  for (var i = 0u; i < 3u; i++) {
    let ax = vec3<f32>(axes[i].x, axes[i].y, axes[i].z);
    let ad = abs(dot(ax, dir));
    if (ad > max_abs) {
      max_abs = ad;
      best = i;
    }
  }
  return best;
}

// ─── Box-Box 3D SAT + Clipping ──────────────────────────────────────────────

fn collide_box_box_3d(idx_a: u32, idx_b: u32, mu: f32) {
  let pos_a = body_pos3(idx_a);
  let pos_b = body_pos3(idx_b);
  let q_a = body_quat(idx_a);
  let q_b = body_quat(idx_b);
  let he_a = collider_half_ext3(idx_a);
  let he_b = collider_half_ext3(idx_b);

  let axes_a = quat_to_axes(q_a);
  let axes_b = quat_to_axes(q_b);

  let he_a_arr = array<f32, 3>(he_a.x, he_a.y, he_a.z);
  let he_b_arr = array<f32, 3>(he_b.x, he_b.y, he_b.z);

  let d = pos_b - pos_a;

  var min_overlap = 1e30f;
  var best_axis = vec3<f32>(0.0, 1.0, 0.0);
  var best_axis_index = 0u;
  var best_axis_type = 0u; // 0 = faceA, 1 = faceB, 2 = edge

  // Helper arrays for axis access
  var ax_a: array<vec3<f32>, 3>;
  var ax_b: array<vec3<f32>, 3>;
  ax_a[0] = axes_a[0];
  ax_a[1] = axes_a[1];
  ax_a[2] = axes_a[2];
  ax_b[0] = axes_b[0];
  ax_b[1] = axes_b[1];
  ax_b[2] = axes_b[2];

  // ─── Test 3 face normals of A ─────────────────────────────────────────
  for (var i = 0u; i < 3u; i++) {
    let axis = ax_a[i];
    var radius_a = 0.0f;
    var radius_b = 0.0f;
    for (var k = 0u; k < 3u; k++) {
      radius_a += abs(dot(ax_a[k], axis)) * he_a_arr[k];
      radius_b += abs(dot(ax_b[k], axis)) * he_b_arr[k];
    }
    let distance = abs(dot(d, axis));
    let overlap = radius_a + radius_b - distance;
    if (overlap < 0.0) { return; }
    if (overlap < min_overlap) {
      min_overlap = overlap;
      best_axis = axis;
      best_axis_index = i;
      best_axis_type = 0u;
    }
  }

  // ─── Test 3 face normals of B ─────────────────────────────────────────
  for (var i = 0u; i < 3u; i++) {
    let axis = ax_b[i];
    var radius_a = 0.0f;
    var radius_b = 0.0f;
    for (var k = 0u; k < 3u; k++) {
      radius_a += abs(dot(ax_a[k], axis)) * he_a_arr[k];
      radius_b += abs(dot(ax_b[k], axis)) * he_b_arr[k];
    }
    let distance = abs(dot(d, axis));
    let overlap = radius_a + radius_b - distance;
    if (overlap < 0.0) { return; }
    if (overlap < min_overlap) {
      min_overlap = overlap;
      best_axis = axis;
      best_axis_index = 3u + i;
      best_axis_type = 1u;
    }
  }

  // ─── Test 9 edge-edge cross products ──────────────────────────────────
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      let c = cross(ax_a[i], ax_b[j]);
      let len = length(c);
      if (len < 1e-6) {
        continue; // Near-parallel edges, skip
      }
      let axis = c / len;
      var radius_a = 0.0f;
      var radius_b = 0.0f;
      for (var k = 0u; k < 3u; k++) {
        radius_a += abs(dot(ax_a[k], axis)) * he_a_arr[k];
        radius_b += abs(dot(ax_b[k], axis)) * he_b_arr[k];
      }
      let distance = abs(dot(d, axis));
      let overlap = radius_a + radius_b - distance;
      if (overlap < 0.0) { return; }
      if (overlap < min_overlap) {
        min_overlap = overlap;
        best_axis = axis;
        best_axis_index = 6u + i * 3u + j;
        best_axis_type = 2u;
      }
    }
  }

  // Ensure normal points from B to A
  if (dot(pos_a - pos_b, best_axis) < 0.0) {
    best_axis = -best_axis;
  }

  // ─── Generate contacts ────────────────────────────────────────────────
  if (best_axis_type == 2u) {
    // Edge-edge: single contact at midpoint
    let midpoint = (pos_a + pos_b) * 0.5;
    let feature_id = best_axis_index;
    emit_contact3(idx_a, idx_b, feature_id, best_axis, min_overlap + COLLISION_MARGIN, midpoint, mu);
    return;
  }

  // Face contact: clip incident face against reference face
  var ref_face: ClipPolygon;
  var inc_face: ClipPolygon;
  var ref_normal: vec3<f32>;
  var ref_axes: mat3x3<f32>;
  var ref_he: vec3<f32>;
  var ref_pos: vec3<f32>;
  var ref_face_axis: u32;

  if (best_axis_type == 0u) {
    // Reference face on body A
    let face_idx = best_axis_index;
    let dir_to_b = -best_axis;
    let sign_val = select(-1.0, 1.0, dot(dir_to_b, ax_a[face_idx]) > 0.0);
    ref_normal = ax_a[face_idx] * sign_val;
    ref_face = get_box_face_verts(pos_a, axes_a, he_a, face_idx, sign_val);
    ref_axes = axes_a;
    ref_he = he_a;
    ref_pos = pos_a;
    ref_face_axis = face_idx;

    // Incident face on B: most anti-aligned with ref_normal
    let inc_axis = find_most_aligned_axis(axes_b, ref_normal);
    let inc_sign = select(-1.0, 1.0, dot(best_axis, ax_b[inc_axis]) > 0.0);
    inc_face = get_box_face_verts(pos_b, axes_b, he_b, inc_axis, inc_sign);
  } else {
    // Reference face on body B
    let face_idx = best_axis_index - 3u;
    let sign_val = select(-1.0, 1.0, dot(best_axis, ax_b[face_idx]) > 0.0);
    ref_normal = ax_b[face_idx] * sign_val;
    ref_face = get_box_face_verts(pos_b, axes_b, he_b, face_idx, sign_val);
    ref_axes = axes_b;
    ref_he = he_b;
    ref_pos = pos_b;
    ref_face_axis = face_idx;

    // Incident face on A
    let dir_to_b = -best_axis;
    let inc_axis = find_most_aligned_axis(axes_a, ref_normal);
    let inc_sign = select(-1.0, 1.0, dot(dir_to_b, ax_a[inc_axis]) > 0.0);
    inc_face = get_box_face_verts(pos_a, axes_a, he_a, inc_axis, inc_sign);
  }

  // Clip incident polygon against 4 side planes of reference face
  var clipped = inc_face;

  let ref_he_arr = array<f32, 3>(ref_he.x, ref_he.y, ref_he.z);
  let a1 = (ref_face_axis + 1u) % 3u;
  let a2 = (ref_face_axis + 2u) % 3u;

  var ref_ax: array<vec3<f32>, 3>;
  ref_ax[0] = ref_axes[0];
  ref_ax[1] = ref_axes[1];
  ref_ax[2] = ref_axes[2];

  // Side plane 1: +tangent1
  let tang1 = ref_ax[a1];
  let center_dot_t1 = dot(ref_pos, tang1);
  clipped = clip_polygon_by_plane(clipped, tang1, center_dot_t1 + ref_he_arr[a1]);
  if (clipped.count == 0u) {
    let mid = (pos_a + pos_b) * 0.5;
    emit_contact3(idx_a, idx_b, best_axis_index, best_axis, min_overlap + COLLISION_MARGIN, mid, mu);
    return;
  }

  // Side plane 2: -tangent1
  clipped = clip_polygon_by_plane(clipped, -tang1, -center_dot_t1 + ref_he_arr[a1]);
  if (clipped.count == 0u) {
    let mid = (pos_a + pos_b) * 0.5;
    emit_contact3(idx_a, idx_b, best_axis_index, best_axis, min_overlap + COLLISION_MARGIN, mid, mu);
    return;
  }

  // Side plane 3: +tangent2
  let tang2 = ref_ax[a2];
  let center_dot_t2 = dot(ref_pos, tang2);
  clipped = clip_polygon_by_plane(clipped, tang2, center_dot_t2 + ref_he_arr[a2]);
  if (clipped.count == 0u) {
    let mid = (pos_a + pos_b) * 0.5;
    emit_contact3(idx_a, idx_b, best_axis_index, best_axis, min_overlap + COLLISION_MARGIN, mid, mu);
    return;
  }

  // Side plane 4: -tangent2
  clipped = clip_polygon_by_plane(clipped, -tang2, -center_dot_t2 + ref_he_arr[a2]);
  if (clipped.count == 0u) {
    let mid = (pos_a + pos_b) * 0.5;
    emit_contact3(idx_a, idx_b, best_axis_index, best_axis, min_overlap + COLLISION_MARGIN, mid, mu);
    return;
  }

  // Filter points behind reference face and emit contacts
  let ref_face_offset = dot(ref_normal, ref_face.verts[0]);
  var emitted = 0u;

  for (var i = 0u; i < clipped.count; i++) {
    let p = clipped.verts[i];
    let sep = dot(ref_normal, p) - ref_face_offset;
    if (sep <= 0.01) {
      let contact_depth = max(0.0, -sep) + COLLISION_MARGIN;
      let feature_id = (best_axis_index << 4u) | i;
      emit_contact3(idx_a, idx_b, feature_id, best_axis, contact_depth, p, mu);
      emitted += 1u;
      if (emitted >= 4u) {
        break; // Max 4 contacts per face-face pair
      }
    }
  }

  if (emitted == 0u) {
    let mid = (pos_a + pos_b) * 0.5;
    emit_contact3(idx_a, idx_b, best_axis_index, best_axis, min_overlap + COLLISION_MARGIN, mid, mu);
  }
}

// ─── Box-Sphere 3D ──────────────────────────────────────────────────────────

fn collide_box_sphere_3d(box_idx: u32, sphere_idx: u32, mu: f32, swap: bool) {
  let box_pos = body_pos3(box_idx);
  let sphere_pos = body_pos3(sphere_idx);
  let q = body_quat(box_idx);
  let he = collider_half_ext3(box_idx);
  let radius = collider_radius3(sphere_idx);

  // Transform sphere center to box local space
  let d = sphere_pos - box_pos;
  let local_center = quat_rotate_inv(q, d);

  // Clamp to box half-extents
  let closest = clamp(local_center, -he, he);
  let diff = local_center - closest;
  let dist_sq = dot(diff, diff);

  if (dist_sq >= radius * radius) {
    return;
  }

  let dist = sqrt(dist_sq);
  var local_normal: vec3<f32>;
  var depth: f32;

  if (dist > 1e-10) {
    local_normal = diff / dist;
    depth = radius - dist;
  } else {
    // Sphere center inside box: push out along closest face
    let dx = he.x - abs(local_center.x);
    let dy = he.y - abs(local_center.y);
    let dz = he.z - abs(local_center.z);

    if (dx <= dy && dx <= dz) {
      local_normal = select(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), local_center.x >= 0.0);
      depth = dx + radius;
    } else if (dy <= dz) {
      local_normal = select(vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), local_center.y >= 0.0);
      depth = dy + radius;
    } else {
      local_normal = select(vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0), local_center.z >= 0.0);
      depth = dz + radius;
    }
  }

  // Transform normal back to world space
  let outward = quat_rotate(q, local_normal);
  // Contact point on sphere surface
  let contact_point = sphere_pos - outward * radius;

  // Convention: normal from B to A
  var manifold_normal: vec3<f32>;
  var out_a: u32;
  var out_b: u32;

  if (swap) {
    manifold_normal = outward;
    out_a = sphere_idx;
    out_b = box_idx;
  } else {
    manifold_normal = -outward;
    out_a = box_idx;
    out_b = sphere_idx;
  }

  let feature_id = 0x100u;
  emit_contact3(out_a, out_b, feature_id, manifold_normal, depth + COLLISION_MARGIN, contact_point, mu);
}

// ─── Sphere-Sphere 3D ───────────────────────────────────────────────────────

fn collide_sphere_sphere_3d(idx_a: u32, idx_b: u32, mu: f32) {
  let pos_a = body_pos3(idx_a);
  let pos_b = body_pos3(idx_b);
  let radius_a = collider_radius3(idx_a);
  let radius_b = collider_radius3(idx_b);

  let d = pos_a - pos_b;
  let dist = length(d);
  let combined = radius_a + radius_b;

  if (dist >= combined) {
    return;
  }

  var normal: vec3<f32>;
  if (dist > 1e-10) {
    normal = d / dist;
  } else {
    normal = vec3<f32>(0.0, 1.0, 0.0);
  }

  let depth = combined - dist + COLLISION_MARGIN;
  let contact_point = pos_b + normal * radius_b;
  let feature_id = 0x200u;
  emit_contact3(idx_a, idx_b, feature_id, normal, depth, contact_point, mu);
}

// ─── Main entry point ───────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn narrowphase_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_idx = gid.x;
  if (pair_idx >= params.num_pairs) {
    return;
  }

  let body_a = pair_buffer[pair_idx * 2u];
  let body_b = pair_buffer[pair_idx * 2u + 1u];
  let shape_a = collider_shape(body_a);
  let shape_b = collider_shape(body_b);

  let friction_a = collider_friction3(body_a);
  let friction_b = collider_friction3(body_b);
  let mu = sqrt(friction_a * friction_b);

  if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_CUBOID) {
    collide_box_box_3d(body_a, body_b, mu);
  } else if (shape_a == SHAPE_BALL && shape_b == SHAPE_BALL) {
    collide_sphere_sphere_3d(body_a, body_b, mu);
  } else if (shape_a == SHAPE_CUBOID && shape_b == SHAPE_BALL) {
    collide_box_sphere_3d(body_a, body_b, mu, false);
  } else if (shape_a == SHAPE_BALL && shape_b == SHAPE_CUBOID) {
    collide_box_sphere_3d(body_b, body_a, mu, true);
  }
}
