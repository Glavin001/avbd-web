// AUTO-GENERATED — do not edit. Run: npx tsx scripts/embed-shaders.ts

export const BVH_BUILD_WGSL = `// Karras 2012 LBVH hierarchy construction.
// Builds the BVH topology (parent/child pointers) from sorted Morton codes.
// Works for both 2D and 3D — topology only, no AABBs.
//
// Index convention:
//   Internal nodes: [0, N-2]   (N-1 internal nodes)
//   Leaf encoding in child arrays:
//     child >= 0  → internal node index
//     child < 0   → leaf index = -(child) - 1   (leaf 0 = -1, leaf 1 = -2, ...)
//   Parent array covers all nodes: [0, 2N-2]
//     parent[0..N-2] = parent of internal node i
//     parent[N-1..2N-2] = parent of leaf (index - (N-1))
//     Actually: parent array size = 2N-1, where:
//       parent[i] for i in [0, N-2] = parent of internal node i
//       parent[N-1 + j] for j in [0, N-1] = parent of leaf j

struct Params {
  num_leaves: u32,
};

@group(0) @binding(0) var<storage, read> morton_codes: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

// Output: topology arrays
// left_child[i], right_child[i] for internal node i (size N-1)
@group(0) @binding(2) var<storage, read_write> left_child: array<i32>;
@group(0) @binding(3) var<storage, read_write> right_child: array<i32>;

// parent[k] for node k in [0, 2N-2]
// k in [0, N-2]: internal node k
// k in [N-1, 2N-2]: leaf (k - N + 1)
@group(0) @binding(4) var<storage, read_write> parent: array<i32>;

// Delta function: longest common prefix length between morton codes at indices i and j.
// Returns -1 if j is out of range. Uses index-based tiebreaker for equal codes.
fn delta(i: i32, j: i32, n: i32) -> i32 {
  if (j < 0 || j >= n) {
    return -1;
  }

  let mi = morton_codes[u32(i)];
  let mj = morton_codes[u32(j)];

  if (mi == mj) {
    // Tiebreaker: use index XOR with +32 offset
    return i32(countLeadingZeros(u32(i) ^ u32(j))) + 32;
  }

  return i32(countLeadingZeros(mi ^ mj));
}

@compute @workgroup_size(256)
fn bvh_build(@builtin(global_invocation_id) gid: vec3u) {
  let idx = i32(gid.x);
  let n = i32(params.num_leaves);
  let num_internal = n - 1;

  if (idx >= num_internal) {
    return;
  }

  // Step 1: Determine direction of the range
  let delta_right = delta(idx, idx + 1, n);
  let delta_left  = delta(idx, idx - 1, n);
  let d = select(-1, 1, delta_right > delta_left);

  // Step 2: Compute upper bound for range length
  let delta_min = delta(idx, idx - d, n);

  var l_max = 2;
  while (delta(idx, idx + l_max * d, n) > delta_min) {
    l_max = l_max * 2;
  }

  // Step 3: Binary search for the other end
  var l = 0;
  var t = l_max / 2;
  while (t >= 1) {
    if (delta(idx, idx + (l + t) * d, n) > delta_min) {
      l = l + t;
    }
    t = t / 2;
  }

  let j = idx + l * d;

  // Step 4: Find split position
  let delta_node = delta(idx, j, n);

  var s = 0;
  var divisor = 2;
  var t2 = i32(ceil(f32(l) / f32(divisor)));

  while (t2 >= 1) {
    if (delta(idx, idx + (s + t2) * d, n) > delta_node) {
      s = s + t2;
    }
    divisor = divisor * 2;
    t2 = i32(ceil(f32(l) / f32(divisor)));
  }

  let split = idx + s * d + min(d, 0);

  // Step 5: Determine children
  let range_min = min(idx, j);
  let range_max = max(idx, j);

  // Left child
  var lc: i32;
  if (split == range_min) {
    // Left child is a leaf
    lc = -(split) - 1;
  } else {
    // Left child is an internal node
    lc = split;
  }

  // Right child
  var rc: i32;
  if (split + 1 == range_max) {
    // Right child is a leaf
    rc = -(split + 1) - 1;
  } else {
    // Right child is an internal node
    rc = split + 1;
  }

  left_child[u32(idx)]  = lc;
  right_child[u32(idx)] = rc;

  // Step 6: Set parent pointers for children
  if (lc >= 0) {
    // Internal node child
    parent[u32(lc)] = idx;
  } else {
    // Leaf child: parent index = (N-1) + leaf_index
    let leaf_idx = -(lc) - 1;
    parent[u32(num_internal + leaf_idx)] = idx;
  }

  if (rc >= 0) {
    parent[u32(rc)] = idx;
  } else {
    let leaf_idx = -(rc) - 1;
    parent[u32(num_internal + leaf_idx)] = idx;
  }

  // Root node (internal node 0) has no parent — set sentinel
  if (idx == 0) {
    parent[0u] = -1;
  }
}
`;

export const BVH_REFIT_2D_WGSL = `// Bottom-up AABB refit for 2D BVH.
// One thread per leaf. Each leaf writes its AABB, then walks up the parent
// chain using atomic visit counts to ensure each internal node is processed
// exactly once (by the second visitor).

struct Params {
  num_leaves:   u32,
  num_internal: u32,  // = num_leaves - 1
};

// Sorted indices: maps leaf i to original body index
@group(0) @binding(0) var<storage, read> sorted_indices: array<u32>;

// Per-body AABBs: 4 floats [minX, minY, maxX, maxY]
@group(0) @binding(1) var<storage, read> aabb_buffer: array<f32>;

// BVH topology
@group(0) @binding(2) var<storage, read> left_child: array<i32>;
@group(0) @binding(3) var<storage, read> right_child: array<i32>;
@group(0) @binding(4) var<storage, read> parent_buf: array<i32>;

@group(0) @binding(5) var<uniform> params: Params;

// Output: node AABBs for all 2N-1 nodes, 4 floats each
// Layout: [internal_0, internal_1, ..., internal_{N-2}, leaf_0, leaf_1, ..., leaf_{N-1}]
@group(0) @binding(6) var<storage, read_write> node_aabb: array<f32>;

// Atomic visit counters for internal nodes (size N-1), initialised to 0
@group(0) @binding(7) var<storage, read_write> visit_count: array<atomic<u32>>;

fn load_node_aabb_min_x(node: u32) -> f32 { return node_aabb[node * 4u + 0u]; }
fn load_node_aabb_min_y(node: u32) -> f32 { return node_aabb[node * 4u + 1u]; }
fn load_node_aabb_max_x(node: u32) -> f32 { return node_aabb[node * 4u + 2u]; }
fn load_node_aabb_max_y(node: u32) -> f32 { return node_aabb[node * 4u + 3u]; }

fn get_node_index_for_child(child: i32, num_internal: u32) -> u32 {
  // child >= 0 → internal node index = child
  // child < 0  → leaf index = -(child)-1, node index = num_internal + leaf_index
  if (child >= 0) {
    return u32(child);
  }
  let leaf_idx = u32(-(child) - 1);
  return num_internal + leaf_idx;
}

@compute @workgroup_size(256)
fn bvh_refit_2d(@builtin(global_invocation_id) gid: vec3u) {
  let leaf_idx = gid.x;
  if (leaf_idx >= params.num_leaves) {
    return;
  }

  let ni = params.num_internal;

  // Step 1: Write leaf AABB
  let body_idx = sorted_indices[leaf_idx];
  let src_base = body_idx * 4u;
  let dst_node = ni + leaf_idx;
  let dst_base = dst_node * 4u;

  node_aabb[dst_base + 0u] = aabb_buffer[src_base + 0u];
  node_aabb[dst_base + 1u] = aabb_buffer[src_base + 1u];
  node_aabb[dst_base + 2u] = aabb_buffer[src_base + 2u];
  node_aabb[dst_base + 3u] = aabb_buffer[src_base + 3u];

  // Step 2: Walk up parent chain
  // Parent index for this leaf: parent_buf[num_internal + leaf_idx]
  var p = parent_buf[ni + leaf_idx];

  while (p >= 0) {
    let pu = u32(p);

    // Atomic increment — first visitor (result 0) exits, second visitor (result 1) continues
    let prev = atomicAdd(&visit_count[pu], 1u);
    if (prev == 0u) {
      // First visitor: the other child hasn't written its AABB yet. Stop.
      return;
    }

    // Second visitor: both children are ready. Compute union AABB.
    let lc = left_child[pu];
    let rc = right_child[pu];

    let ln = get_node_index_for_child(lc, ni);
    let rn = get_node_index_for_child(rc, ni);

    let union_min_x = min(load_node_aabb_min_x(ln), load_node_aabb_min_x(rn));
    let union_min_y = min(load_node_aabb_min_y(ln), load_node_aabb_min_y(rn));
    let union_max_x = max(load_node_aabb_max_x(ln), load_node_aabb_max_x(rn));
    let union_max_y = max(load_node_aabb_max_y(ln), load_node_aabb_max_y(rn));

    let out_base = pu * 4u;
    node_aabb[out_base + 0u] = union_min_x;
    node_aabb[out_base + 1u] = union_min_y;
    node_aabb[out_base + 2u] = union_max_x;
    node_aabb[out_base + 3u] = union_max_y;

    // Move to parent's parent
    p = parent_buf[pu];
  }
}
`;

export const BVH_REFIT_3D_WGSL = `// Bottom-up AABB refit for 3D BVH.
// Same algorithm as 2D but with 6-float AABBs (minX,minY,minZ, maxX,maxY,maxZ).

struct Params {
  num_leaves:   u32,
  num_internal: u32,  // = num_leaves - 1
};

// Sorted indices: maps leaf i to original body index
@group(0) @binding(0) var<storage, read> sorted_indices: array<u32>;

// Per-body AABBs: 6 floats [minX, minY, minZ, maxX, maxY, maxZ]
@group(0) @binding(1) var<storage, read> aabb_buffer: array<f32>;

// BVH topology
@group(0) @binding(2) var<storage, read> left_child: array<i32>;
@group(0) @binding(3) var<storage, read> right_child: array<i32>;
@group(0) @binding(4) var<storage, read> parent_buf: array<i32>;

@group(0) @binding(5) var<uniform> params: Params;

// Output: node AABBs for all 2N-1 nodes, 6 floats each
// Layout: [internal_0, ..., internal_{N-2}, leaf_0, ..., leaf_{N-1}]
@group(0) @binding(6) var<storage, read_write> node_aabb: array<f32>;

// Atomic visit counters for internal nodes (size N-1), initialised to 0
@group(0) @binding(7) var<storage, read_write> visit_count: array<atomic<u32>>;

const STRIDE: u32 = 6u;

fn load_node_aabb(node: u32, offset: u32) -> f32 {
  return node_aabb[node * STRIDE + offset];
}

fn get_node_index_for_child(child: i32, num_internal: u32) -> u32 {
  if (child >= 0) {
    return u32(child);
  }
  let leaf_idx = u32(-(child) - 1);
  return num_internal + leaf_idx;
}

@compute @workgroup_size(256)
fn bvh_refit_3d(@builtin(global_invocation_id) gid: vec3u) {
  let leaf_idx = gid.x;
  if (leaf_idx >= params.num_leaves) {
    return;
  }

  let ni = params.num_internal;

  // Step 1: Write leaf AABB
  let body_idx = sorted_indices[leaf_idx];
  let src_base = body_idx * STRIDE;
  let dst_node = ni + leaf_idx;
  let dst_base = dst_node * STRIDE;

  node_aabb[dst_base + 0u] = aabb_buffer[src_base + 0u];
  node_aabb[dst_base + 1u] = aabb_buffer[src_base + 1u];
  node_aabb[dst_base + 2u] = aabb_buffer[src_base + 2u];
  node_aabb[dst_base + 3u] = aabb_buffer[src_base + 3u];
  node_aabb[dst_base + 4u] = aabb_buffer[src_base + 4u];
  node_aabb[dst_base + 5u] = aabb_buffer[src_base + 5u];

  // Step 2: Walk up parent chain
  var p = parent_buf[ni + leaf_idx];

  while (p >= 0) {
    let pu = u32(p);

    // Atomic increment — first visitor exits, second visitor continues
    let prev = atomicAdd(&visit_count[pu], 1u);
    if (prev == 0u) {
      return;
    }

    // Second visitor: compute union of children AABBs
    let lc = left_child[pu];
    let rc = right_child[pu];

    let ln = get_node_index_for_child(lc, ni);
    let rn = get_node_index_for_child(rc, ni);

    let union_min_x = min(load_node_aabb(ln, 0u), load_node_aabb(rn, 0u));
    let union_min_y = min(load_node_aabb(ln, 1u), load_node_aabb(rn, 1u));
    let union_min_z = min(load_node_aabb(ln, 2u), load_node_aabb(rn, 2u));
    let union_max_x = max(load_node_aabb(ln, 3u), load_node_aabb(rn, 3u));
    let union_max_y = max(load_node_aabb(ln, 4u), load_node_aabb(rn, 4u));
    let union_max_z = max(load_node_aabb(ln, 5u), load_node_aabb(rn, 5u));

    let out_base = pu * STRIDE;
    node_aabb[out_base + 0u] = union_min_x;
    node_aabb[out_base + 1u] = union_min_y;
    node_aabb[out_base + 2u] = union_min_z;
    node_aabb[out_base + 3u] = union_max_x;
    node_aabb[out_base + 4u] = union_max_y;
    node_aabb[out_base + 5u] = union_max_z;

    // Move to parent's parent
    p = parent_buf[pu];
  }
}
`;

export const BVH_TRAVERSE_2D_WGSL = `// BVH traversal for 2D broad-phase pair finding.
// One thread per body. Uses a fixed-size stack to traverse the BVH and
// output overlapping AABB pairs.

struct Params {
  num_bodies:   u32,
  num_internal: u32,  // = num_bodies - 1
  max_pairs:    u32,
  _pad0:        u32,
};

// Node AABBs: (2N-1) nodes, 4 floats each [minX, minY, maxX, maxY]
// Layout: [internal_0..internal_{N-2}, leaf_0..leaf_{N-1}]
@group(0) @binding(0) var<storage, read> node_aabb: array<f32>;

// BVH topology (N-1 internal nodes)
@group(0) @binding(1) var<storage, read> left_child: array<i32>;
@group(0) @binding(2) var<storage, read> right_child: array<i32>;

// Per-body AABBs: 4 floats each (original body order, not sorted)
@group(0) @binding(3) var<storage, read> aabb_buffer: array<f32>;

// Sorted indices: maps leaf index to original body index
@group(0) @binding(4) var<storage, read> sorted_indices: array<u32>;

// Body types: 0 = Dynamic, 1 = Fixed, etc.
@group(0) @binding(5) var<storage, read> body_types: array<u32>;

@group(0) @binding(6) var<uniform> params: Params;

// Output: pair buffer [bodyA, bodyB] pairs, and atomic pair count
@group(0) @binding(7) var<storage, read_write> pair_buffer: array<u32>;
@group(0) @binding(8) var<storage, read_write> pair_count: array<atomic<u32>>;

const MAX_STACK_DEPTH: u32 = 32u;
const AABB_STRIDE: u32 = 4u;

fn aabb_overlap_2d(
  a_min_x: f32, a_min_y: f32, a_max_x: f32, a_max_y: f32,
  b_min_x: f32, b_min_y: f32, b_max_x: f32, b_max_y: f32
) -> bool {
  return !(a_max_x < b_min_x || a_min_x > b_max_x ||
           a_max_y < b_min_y || a_min_y > b_max_y);
}

@compute @workgroup_size(256)
fn bvh_traverse_2d(@builtin(global_invocation_id) gid: vec3u) {
  let body_i = gid.x;
  if (body_i >= params.num_bodies) {
    return;
  }

  let ni = params.num_internal;

  // Load AABB for body i (in original body order)
  let a_base = body_i * AABB_STRIDE;
  let a_min_x = aabb_buffer[a_base + 0u];
  let a_min_y = aabb_buffer[a_base + 1u];
  let a_max_x = aabb_buffer[a_base + 2u];
  let a_max_y = aabb_buffer[a_base + 3u];

  let type_i = body_types[body_i];

  // Stack-based traversal
  var stack: array<i32, 32>;  // encoded the same as children: >=0 internal, <0 leaf
  var stack_top: u32 = 0u;

  // Push root (internal node 0) — only if there are internal nodes
  if (ni > 0u) {
    stack[0] = 0;  // internal node 0
    stack_top = 1u;
  }

  while (stack_top > 0u) {
    stack_top -= 1u;
    let node = stack[stack_top];

    if (node >= 0) {
      // Internal node — load its AABB and test overlap
      let n_base = u32(node) * AABB_STRIDE;
      let n_min_x = node_aabb[n_base + 0u];
      let n_min_y = node_aabb[n_base + 1u];
      let n_max_x = node_aabb[n_base + 2u];
      let n_max_y = node_aabb[n_base + 3u];

      if (aabb_overlap_2d(a_min_x, a_min_y, a_max_x, a_max_y,
                          n_min_x, n_min_y, n_max_x, n_max_y)) {
        // Push both children
        let lc = left_child[u32(node)];
        let rc = right_child[u32(node)];

        if (stack_top < MAX_STACK_DEPTH) {
          stack[stack_top] = lc;
          stack_top += 1u;
        }
        if (stack_top < MAX_STACK_DEPTH) {
          stack[stack_top] = rc;
          stack_top += 1u;
        }
      }
    } else {
      // Leaf node
      let leaf_idx = u32(-(node) - 1);
      let body_j = sorted_indices[leaf_idx];

      // Skip self
      if (body_j == body_i) {
        continue;
      }

      // Deduplication: only emit pair where body_i < body_j
      if (body_i >= body_j) {
        continue;
      }

      // Skip if both bodies are fixed/static
      let type_j = body_types[body_j];
      if (type_i >= 1u && type_j >= 1u) {
        continue;
      }

      // Load leaf AABB from node_aabb (leaf nodes stored after internal nodes)
      let l_node = ni + leaf_idx;
      let l_base = l_node * AABB_STRIDE;
      let l_min_x = node_aabb[l_base + 0u];
      let l_min_y = node_aabb[l_base + 1u];
      let l_max_x = node_aabb[l_base + 2u];
      let l_max_y = node_aabb[l_base + 3u];

      if (aabb_overlap_2d(a_min_x, a_min_y, a_max_x, a_max_y,
                          l_min_x, l_min_y, l_max_x, l_max_y)) {
        // Emit pair
        let pair_idx = atomicAdd(&pair_count[0], 1u);
        if (pair_idx < params.max_pairs) {
          pair_buffer[pair_idx * 2u + 0u] = body_i;
          pair_buffer[pair_idx * 2u + 1u] = body_j;
        }
      }
    }
  }
}
`;

export const BVH_TRAVERSE_3D_WGSL = `// BVH traversal for 3D broad-phase pair finding.
// One thread per body. Uses a fixed-size stack to traverse the BVH and
// output overlapping AABB pairs.

struct Params {
  num_bodies:   u32,
  num_internal: u32,  // = num_bodies - 1
  max_pairs:    u32,
  _pad0:        u32,
};

// Node AABBs: (2N-1) nodes, 6 floats each [minX, minY, minZ, maxX, maxY, maxZ]
// Layout: [internal_0..internal_{N-2}, leaf_0..leaf_{N-1}]
@group(0) @binding(0) var<storage, read> node_aabb: array<f32>;

// BVH topology (N-1 internal nodes)
@group(0) @binding(1) var<storage, read> left_child: array<i32>;
@group(0) @binding(2) var<storage, read> right_child: array<i32>;

// Per-body AABBs: 6 floats each (original body order, not sorted)
@group(0) @binding(3) var<storage, read> aabb_buffer: array<f32>;

// Sorted indices: maps leaf index to original body index
@group(0) @binding(4) var<storage, read> sorted_indices: array<u32>;

// Body types: 0 = Dynamic, 1 = Fixed, etc.
@group(0) @binding(5) var<storage, read> body_types: array<u32>;

@group(0) @binding(6) var<uniform> params: Params;

// Output: pair buffer [bodyA, bodyB] pairs, and atomic pair count
@group(0) @binding(7) var<storage, read_write> pair_buffer: array<u32>;
@group(0) @binding(8) var<storage, read_write> pair_count: array<atomic<u32>>;

const MAX_STACK_DEPTH: u32 = 32u;
const AABB_STRIDE: u32 = 6u;

fn aabb_overlap_3d(
  a_min_x: f32, a_min_y: f32, a_min_z: f32,
  a_max_x: f32, a_max_y: f32, a_max_z: f32,
  b_min_x: f32, b_min_y: f32, b_min_z: f32,
  b_max_x: f32, b_max_y: f32, b_max_z: f32
) -> bool {
  return !(a_max_x < b_min_x || a_min_x > b_max_x ||
           a_max_y < b_min_y || a_min_y > b_max_y ||
           a_max_z < b_min_z || a_min_z > b_max_z);
}

@compute @workgroup_size(256)
fn bvh_traverse_3d(@builtin(global_invocation_id) gid: vec3u) {
  let body_i = gid.x;
  if (body_i >= params.num_bodies) {
    return;
  }

  let ni = params.num_internal;

  // Load AABB for body i (in original body order)
  let a_base = body_i * AABB_STRIDE;
  let a_min_x = aabb_buffer[a_base + 0u];
  let a_min_y = aabb_buffer[a_base + 1u];
  let a_min_z = aabb_buffer[a_base + 2u];
  let a_max_x = aabb_buffer[a_base + 3u];
  let a_max_y = aabb_buffer[a_base + 4u];
  let a_max_z = aabb_buffer[a_base + 5u];

  let type_i = body_types[body_i];

  // Stack-based traversal
  var stack: array<i32, 32>;
  var stack_top: u32 = 0u;

  // Push root (internal node 0) — only if there are internal nodes
  if (ni > 0u) {
    stack[0] = 0;
    stack_top = 1u;
  }

  while (stack_top > 0u) {
    stack_top -= 1u;
    let node = stack[stack_top];

    if (node >= 0) {
      // Internal node — load its AABB and test overlap
      let n_base = u32(node) * AABB_STRIDE;
      let n_min_x = node_aabb[n_base + 0u];
      let n_min_y = node_aabb[n_base + 1u];
      let n_min_z = node_aabb[n_base + 2u];
      let n_max_x = node_aabb[n_base + 3u];
      let n_max_y = node_aabb[n_base + 4u];
      let n_max_z = node_aabb[n_base + 5u];

      if (aabb_overlap_3d(a_min_x, a_min_y, a_min_z, a_max_x, a_max_y, a_max_z,
                          n_min_x, n_min_y, n_min_z, n_max_x, n_max_y, n_max_z)) {
        let lc = left_child[u32(node)];
        let rc = right_child[u32(node)];

        if (stack_top < MAX_STACK_DEPTH) {
          stack[stack_top] = lc;
          stack_top += 1u;
        }
        if (stack_top < MAX_STACK_DEPTH) {
          stack[stack_top] = rc;
          stack_top += 1u;
        }
      }
    } else {
      // Leaf node
      let leaf_idx = u32(-(node) - 1);
      let body_j = sorted_indices[leaf_idx];

      // Skip self
      if (body_j == body_i) {
        continue;
      }

      // Deduplication: only emit pair where body_i < body_j
      if (body_i >= body_j) {
        continue;
      }

      // Skip if both bodies are fixed/static
      let type_j = body_types[body_j];
      if (type_i >= 1u && type_j >= 1u) {
        continue;
      }

      // Load leaf AABB from node_aabb
      let l_node = ni + leaf_idx;
      let l_base = l_node * AABB_STRIDE;
      let l_min_x = node_aabb[l_base + 0u];
      let l_min_y = node_aabb[l_base + 1u];
      let l_min_z = node_aabb[l_base + 2u];
      let l_max_x = node_aabb[l_base + 3u];
      let l_max_y = node_aabb[l_base + 4u];
      let l_max_z = node_aabb[l_base + 5u];

      if (aabb_overlap_3d(a_min_x, a_min_y, a_min_z, a_max_x, a_max_y, a_max_z,
                          l_min_x, l_min_y, l_min_z, l_max_x, l_max_y, l_max_z)) {
        let pair_idx = atomicAdd(&pair_count[0], 1u);
        if (pair_idx < params.max_pairs) {
          pair_buffer[pair_idx * 2u + 0u] = body_i;
          pair_buffer[pair_idx * 2u + 1u] = body_j;
        }
      }
    }
  }
}
`;

export const CONSTRAINT_ASSEMBLY_2D_WGSL = `// ─── 2D Constraint Assembly ──────────────────────────────────────────────────
// Converts contact points from narrow phase into solver constraint rows.
// Each contact generates 2 rows: normal (non-penetration) + friction (tangent).
//
// Contact input format (8 u32s per contact):
//   [bodyA, bodyB, featureId, _pad, normal_x, normal_y, depth, mu]
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
const CONTACT_STRIDE: u32 = 8u;
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

  // Check if this contact is valid (bodyA != 0xFFFFFFFF sentinel)
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

  // Compute contact point (midpoint along normal at depth)
  // Contact point approximation: posA + depth/2 * normal
  // Actually, we use the body positions directly for the lever arm
  // The contact point is between the bodies along the normal
  let cpx = (posA_x + posB_x) * 0.5;
  let cpy = (posA_y + posB_y) * 0.5;

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
`;

export const CONSTRAINT_ASSEMBLY_3D_WGSL = `// ─── 3D Constraint Assembly ──────────────────────────────────────────────────
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
`;

export const MORTON_CODES_2D_WGSL = `// Morton code generation for 2D LBVH construction.
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
`;

export const MORTON_CODES_3D_WGSL = `// Morton code generation for 3D LBVH construction.
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
`;

export const NARROWPHASE_2D_WGSL = `// ─── AVBD 2D Narrow Phase Compute Shader ─────────────────────────────────────
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
`;

export const NARROWPHASE_3D_WGSL = `// ─── AVBD 3D Narrow Phase Compute Shader ─────────────────────────────────────
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
`;

export const DUAL_UPDATE_3D_WGSL = `// ─── AVBD 3D Dual Update Compute Shader ────────────────────────────────────
// Updates lambda (Lagrange multiplier) and ramps penalty for each constraint.
// Each thread handles one constraint row.
// 3D variant: uses 20-float body state stride, 14-float prev stride,
// and 6-DOF constraint evaluation with quaternion angular displacement.
// Friction coupling: normal contact rows (index % 3 == 0) update adjacent
// two friction tangent rows.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  gravity_z: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  alpha: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
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

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> constraints: array<ConstraintRow3D>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_constraints) { return; }

  var cr = constraints[idx];
  if (cr.is_active == 0u) { return; }

  // Re-evaluate linearized constraint: C = C0*(1-alpha) + J_A·dp_A + J_B·dp_B
  var c_eval = cr.c0 * (1.0 - params.alpha);

  if (cr.body_a >= 0) {
    let ba = u32(cr.body_a) * 20u;
    let bap = u32(cr.body_a) * 14u;
    // Linear displacement
    let dpx = body_state[ba + 0u] - body_prev[bap + 0u];
    let dpy = body_state[ba + 1u] - body_prev[bap + 1u];
    let dpz = body_state[ba + 2u] - body_prev[bap + 2u];
    c_eval += cr.jacobian_a_lin.x * dpx + cr.jacobian_a_lin.y * dpy + cr.jacobian_a_lin.z * dpz;
    // Angular displacement via quaternion difference
    let qw = body_state[ba + 3u]; let qx = body_state[ba + 4u];
    let qy = body_state[ba + 5u]; let qz = body_state[ba + 6u];
    let pqw = body_prev[bap + 3u]; let pqx = body_prev[bap + 4u];
    let pqy = body_prev[bap + 5u]; let pqz = body_prev[bap + 6u];
    // dq = q * conj(q_prev): small angle -> dtheta ≈ 2 * vec3(dq.xyz)
    let dqw = pqw * qw + pqx * qx + pqy * qy + pqz * qz;
    let dqx_v = pqw * qx - pqx * qw - pqy * qz + pqz * qy;
    let dqy_v = pqw * qy + pqx * qz - pqy * qw - pqz * qx;
    let dqz_v = pqw * qz - pqx * qy + pqy * qx - pqz * qw;
    c_eval += cr.jacobian_a_ang.x * 2.0 * dqx_v
           + cr.jacobian_a_ang.y * 2.0 * dqy_v
           + cr.jacobian_a_ang.z * 2.0 * dqz_v;
  }

  if (cr.body_b >= 0) {
    let bb = u32(cr.body_b) * 20u;
    let bbp = u32(cr.body_b) * 14u;
    let dpx = body_state[bb + 0u] - body_prev[bbp + 0u];
    let dpy = body_state[bb + 1u] - body_prev[bbp + 1u];
    let dpz = body_state[bb + 2u] - body_prev[bbp + 2u];
    c_eval += cr.jacobian_b_lin.x * dpx + cr.jacobian_b_lin.y * dpy + cr.jacobian_b_lin.z * dpz;
    let qw = body_state[bb + 3u]; let qx = body_state[bb + 4u];
    let qy = body_state[bb + 5u]; let qz = body_state[bb + 6u];
    let pqw = body_prev[bbp + 3u]; let pqx = body_prev[bbp + 4u];
    let pqy = body_prev[bbp + 5u]; let pqz = body_prev[bbp + 6u];
    let dqw = pqw * qw + pqx * qx + pqy * qy + pqz * qz;
    let dqx_v = pqw * qx - pqx * qw - pqy * qz + pqz * qy;
    let dqy_v = pqw * qy + pqx * qz - pqy * qw - pqz * qx;
    let dqz_v = pqw * qz - pqx * qy + pqy * qx - pqz * qw;
    c_eval += cr.jacobian_b_ang.x * 2.0 * dqx_v
           + cr.jacobian_b_ang.y * 2.0 * dqy_v
           + cr.jacobian_b_ang.z * 2.0 * dqz_v;
  }

  // Stiffness guard: soft constraints zero lambda before accumulation
  var prev_lambda = cr.lambda;
  if (cr.stiffness < 1e30) {
    prev_lambda = 0.0;
  }

  // Update lambda
  var new_lambda = cr.penalty * c_eval + prev_lambda;
  new_lambda = clamp(new_lambda, cr.fmin, cr.fmax);
  cr.lambda = new_lambda;

  // Conditional penalty ramp: only when constraint is interior (not at bounds).
  // For normal contacts: ramp when active. For friction: ramp when not sliding.
  // Reference: manifold.cpp — penalty += beta * |C| when active/sticking
  if (cr.lambda > cr.fmin && cr.lambda < cr.fmax) {
    cr.penalty += params.beta * abs(c_eval);
  }
  cr.penalty = clamp(cr.penalty, params.penalty_min, params.penalty_max);
  if (cr.penalty > cr.stiffness) {
    cr.penalty = cr.stiffness;
  }

  // Write back this row
  constraints[idx] = cr;

  // NOTE: Friction coupling runs as a SEPARATE compute pass (friction-coupling.wgsl)
  // after the dual update, guaranteeing all normal lambdas are finalized.
}
`;

export const DUAL_UPDATE_WGSL = `// ─── AVBD Dual Update Compute Shader ────────────────────────────────────────
// Updates lambda (Lagrange multiplier) and ramps penalty for each constraint.
// Each thread handles one constraint row.
// Includes friction coupling: normal contact rows update adjacent friction rows.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  alpha: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
  _pad: u32,
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

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> constraints: array<ConstraintRow>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_constraints) { return; }

  var cr = constraints[idx];
  if (cr.is_active == 0u) { return; }

  // Re-evaluate linearized constraint: C = C0*(1-alpha) + J*dp
  var c_eval = cr.c0 * (1.0 - params.alpha);
  if (cr.body_a >= 0) {
    let ba_base = u32(cr.body_a) * 8u;
    c_eval += cr.jacobian_a.x * (body_state[ba_base + 0u] - body_prev[ba_base + 0u])
           + cr.jacobian_a.y * (body_state[ba_base + 1u] - body_prev[ba_base + 1u])
           + cr.jacobian_a.z * (body_state[ba_base + 2u] - body_prev[ba_base + 2u]);
  }
  if (cr.body_b >= 0) {
    let bb_base = u32(cr.body_b) * 8u;
    c_eval += cr.jacobian_b.x * (body_state[bb_base + 0u] - body_prev[bb_base + 0u])
           + cr.jacobian_b.y * (body_state[bb_base + 1u] - body_prev[bb_base + 1u])
           + cr.jacobian_b.z * (body_state[bb_base + 2u] - body_prev[bb_base + 2u]);
  }

  // Stiffness guard: soft constraints zero lambda before accumulation
  var prev_lambda = cr.lambda;
  if (cr.stiffness < 1e30) {
    prev_lambda = 0.0;
  }

  // Update lambda
  var new_lambda = cr.penalty * c_eval + prev_lambda;
  new_lambda = clamp(new_lambda, cr.fmin, cr.fmax);
  cr.lambda = new_lambda;

  // Conditional penalty ramp: only when constraint is interior (not at bounds).
  // For normal contacts: ramp when active. For friction: ramp when not sliding.
  // Reference: manifold.cpp — penalty += beta * |C| when active/sticking
  if (cr.lambda > cr.fmin && cr.lambda < cr.fmax) {
    cr.penalty += params.beta * abs(c_eval);
  }
  cr.penalty = clamp(cr.penalty, params.penalty_min, params.penalty_max);
  if (cr.penalty > cr.stiffness) {
    cr.penalty = cr.stiffness;
  }

  // Write back this row
  constraints[idx] = cr;

  // NOTE: Friction coupling runs as a SEPARATE compute pass (friction-coupling.wgsl)
  // after the dual update, guaranteeing all normal lambdas are finalized.
}
`;

export const FRICTION_COUPLING_3D_WGSL = `// ─── AVBD 3D Friction Coupling Compute Shader ───────────────────────────────
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
`;

export const FRICTION_COUPLING_WGSL = `// ─── AVBD Friction Coupling Compute Shader ──────────────────────────────────
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
`;

export const MATH_UTILS_WGSL = `// ─── AVBD Math Utilities for WGSL ───────────────────────────────────────────
// Provides matrix operations for 3x3 (2D) and 6x6 (3D) local solves.

// ─── 3x3 LDL^T Solver (for 2D mode: 3-DOF per body) ────────────────────────

// Solve A * x = b where A is a 3x3 SPD matrix.
// A is stored as mat3x3<f32> (column-major).
// Returns the solution x as a vec3<f32>.
fn solve_ldl3(A: mat3x3<f32>, b: vec3<f32>) -> vec3<f32> {
  // LDL^T decomposition
  // Column 0
  let D0 = A[0][0];
  let L10 = A[0][1] / D0;
  let L20 = A[0][2] / D0;

  // Column 1
  let D1 = A[1][1] - L10 * L10 * D0;
  let L21 = (A[1][2] - L20 * L10 * D0) / D1;

  // Column 2
  let D2 = A[2][2] - L20 * L20 * D0 - L21 * L21 * D1;

  // Forward substitution: L * y = b
  let y0 = b.x;
  let y1 = b.y - L10 * y0;
  let y2 = b.z - L20 * y0 - L21 * y1;

  // Diagonal solve: D * z = y
  let z0 = y0 / D0;
  let z1 = y1 / D1;
  let z2 = y2 / D2;

  // Back substitution: L^T * x = z
  let x2 = z2;
  let x1 = z1 - L21 * x2;
  let x0 = z0 - L10 * x1 - L20 * x2;

  return vec3<f32>(x0, x1, x2);
}

// Outer product of two vec3: result = a * b^T
fn outer_product3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(
    a * b.x,  // column 0
    a * b.y,  // column 1
    a * b.z,  // column 2
  );
}

// Diagonal matrix from absolute values of a vec3 scaled by a scalar
fn diag_abs_scaled(v: vec3<f32>, s: f32) -> mat3x3<f32> {
  return mat3x3<f32>(
    vec3<f32>(abs(v.x) * s, 0.0, 0.0),
    vec3<f32>(0.0, abs(v.y) * s, 0.0),
    vec3<f32>(0.0, 0.0, abs(v.z) * s),
  );
}

// ─── 6x6 LDL^T Solver (for 3D mode: 6-DOF per body) ────────────────────────
// Stored as array<f32, 36> in column-major order.

fn mat6_get(M: array<f32, 36>, row: u32, col: u32) -> f32 {
  return M[col * 6u + row];
}

fn solve_ldl6(A: array<f32, 36>, b: array<f32, 6>) -> array<f32, 6> {
  // L is unit lower triangular, stored below diagonal
  var L: array<f32, 36>;
  var D: array<f32, 6>;

  // Forward pass: compute L and D
  for (var j = 0u; j < 6u; j++) {
    // Compute D[j]
    var sum_d = mat6_get(A, j, j);
    for (var k = 0u; k < j; k++) {
      let ljk = L[k * 6u + j];
      sum_d -= ljk * ljk * D[k];
    }
    D[j] = sum_d;

    // Compute L[i][j] for i > j
    for (var i = j + 1u; i < 6u; i++) {
      var sum_l = mat6_get(A, i, j);
      for (var k = 0u; k < j; k++) {
        sum_l -= L[k * 6u + i] * L[k * 6u + j] * D[k];
      }
      L[j * 6u + i] = sum_l / D[j];
    }
  }

  // Forward substitution: L * y = b
  var y: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) {
    var sum_y = b[i];
    for (var k = 0u; k < i; k++) {
      sum_y -= L[k * 6u + i] * y[k];
    }
    y[i] = sum_y;
  }

  // Diagonal solve: D * z = y
  var z: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) {
    z[i] = y[i] / D[i];
  }

  // Back substitution: L^T * x = z
  var x: array<f32, 6>;
  for (var ii = 0u; ii < 6u; ii++) {
    let i = 5u - ii;
    var sum_x = z[i];
    for (var k = i + 1u; k < 6u; k++) {
      sum_x -= L[i * 6u + k] * x[k];
    }
    x[i] = sum_x;
  }

  return x;
}
`;

export const PRIMAL_UPDATE_2D_WGSL = `// ─── AVBD 2D Primal Update Compute Shader ───────────────────────────────────
// Processes one color group of bodies per dispatch.
// Each thread handles one body: builds 3x3 SPD system, solves via LDL^T.
// Bodies in the same color group share NO constraints → safe parallel update.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  alpha: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
  _pad: u32,
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

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read_write> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read> constraints: array<ConstraintRow>;
@group(0) @binding(4) var<storage, read> color_body_indices: array<u32>;
@group(0) @binding(5) var<storage, read> body_constraint_ranges: array<u32>;
@group(0) @binding(6) var<storage, read> constraint_indices: array<u32>;

fn solve_ldl3(A: mat3x3<f32>, b: vec3<f32>) -> vec3<f32> {
  // Regularize to prevent f32 catastrophic cancellation when penalty >> mass/dt²
  let max_diag = max(A[0][0], max(A[1][1], A[2][2]));
  let eps = 1e-6 * max_diag;
  let D0 = A[0][0] + eps;
  if (D0 <= 0.0) { return vec3<f32>(0.0); }
  let L10 = A[0][1] / D0;
  let L20 = A[0][2] / D0;
  let D1 = A[1][1] + eps - L10 * L10 * D0;
  if (D1 <= 0.0) { return vec3<f32>(0.0); }
  let L21 = (A[1][2] - L20 * L10 * D0) / D1;
  let D2 = A[2][2] + eps - L20 * L20 * D0 - L21 * L21 * D1;
  if (D2 <= 0.0) { return vec3<f32>(0.0); }

  let y0 = b.x;
  let y1 = b.y - L10 * y0;
  let y2 = b.z - L20 * y0 - L21 * y1;

  let x2 = y2 / D2;
  let x1 = y1 / D1 - L21 * x2;
  let x0 = y0 / D0 - L10 * x1 - L20 * x2;

  return vec3<f32>(x0, x1, x2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let thread_id = gid.x;

  // Use num_bodies_in_group from uniform (not arrayLength which returns buffer capacity)
  if (thread_id >= params.num_bodies_in_group) { return; }

  let body_idx = color_body_indices[thread_id];
  let dt = params.dt;
  let dt2 = dt * dt;

  let base = body_idx * 8u;
  let x = body_state[base + 0u];
  let y = body_state[base + 1u];
  let angle = body_state[base + 2u];
  let mass = body_state[base + 6u];
  let inertia = body_state[base + 7u];

  if (mass <= 0.0) { return; }

  let prev_base = body_idx * 8u;
  let prev_x = body_prev[prev_base + 0u];
  let prev_y = body_prev[prev_base + 1u];
  let prev_angle = body_prev[prev_base + 2u];
  let inertial_x = body_prev[prev_base + 3u];
  let inertial_y = body_prev[prev_base + 4u];
  let inertial_angle = body_prev[prev_base + 5u];

  var lhs = mat3x3<f32>(
    vec3<f32>(mass / dt2, 0.0, 0.0),
    vec3<f32>(0.0, mass / dt2, 0.0),
    vec3<f32>(0.0, 0.0, inertia / dt2),
  );

  var rhs = vec3<f32>(
    mass / dt2 * (x - inertial_x),
    mass / dt2 * (y - inertial_y),
    inertia / dt2 * (angle - inertial_angle),
  );

  let range_base = body_idx * 2u;
  let constraint_start = body_constraint_ranges[range_base + 0u];
  let constraint_count = body_constraint_ranges[range_base + 1u];

  for (var ci = 0u; ci < constraint_count; ci++) {
    let cr_idx = constraint_indices[constraint_start + ci];
    let cr = constraints[cr_idx];

    if (cr.is_active == 0u) { continue; }

    var J: vec3<f32>;
    var H_diag: vec3<f32>;
    if (cr.body_a == i32(body_idx)) {
      J = cr.jacobian_a.xyz;
      H_diag = cr.hessian_diag_a.xyz;
    } else {
      J = cr.jacobian_b.xyz;
      H_diag = cr.hessian_diag_b.xyz;
    }

    // Evaluate linearized constraint: C = C0*(1-alpha) + J_A·dp_A + J_B·dp_B
    var c_eval = cr.c0 * (1.0 - params.alpha);
    if (cr.body_a >= 0) {
      let ba_base = u32(cr.body_a) * 8u;
      c_eval += cr.jacobian_a.x * (body_state[ba_base + 0u] - body_prev[ba_base + 0u])
             + cr.jacobian_a.y * (body_state[ba_base + 1u] - body_prev[ba_base + 1u])
             + cr.jacobian_a.z * (body_state[ba_base + 2u] - body_prev[ba_base + 2u]);
    }
    if (cr.body_b >= 0) {
      let bb_base = u32(cr.body_b) * 8u;
      c_eval += cr.jacobian_b.x * (body_state[bb_base + 0u] - body_prev[bb_base + 0u])
             + cr.jacobian_b.y * (body_state[bb_base + 1u] - body_prev[bb_base + 1u])
             + cr.jacobian_b.z * (body_state[bb_base + 2u] - body_prev[bb_base + 2u]);
    }

    // Stiffness guard: soft constraints (finite stiffness) use lambda=0 in primal
    var lambda_for_primal = cr.lambda;
    if (cr.stiffness < 1e30) {
      lambda_for_primal = 0.0;
    }

    var f = cr.penalty * c_eval + lambda_for_primal;
    f = clamp(f, cr.fmin, cr.fmax);

    rhs += J * f;

    let col0 = J * J.x * cr.penalty;
    let col1 = J * J.y * cr.penalty;
    let col2 = J * J.z * cr.penalty;
    lhs[0] += col0;
    lhs[1] += col1;
    lhs[2] += col2;

    let abs_f = abs(f);
    lhs[0].x += abs(H_diag.x) * abs_f;
    lhs[1].y += abs(H_diag.y) * abs_f;
    lhs[2].z += abs(H_diag.z) * abs_f;
  }

  if (lhs[0].x <= 0.0 || lhs[1].y <= 0.0 || lhs[2].z <= 0.0) { return; }

  let delta = solve_ldl3(lhs, rhs);

  body_state[base + 0u] = x - delta.x;
  body_state[base + 1u] = y - delta.y;
  body_state[base + 2u] = angle - delta.z;
}
`;

export const PRIMAL_UPDATE_3D_WGSL = `// ─── AVBD 3D Primal Update Compute Shader ───────────────────────────────────
// 6-DOF per body: position (x,y,z) + angular correction (wx,wy,wz)
// Builds a 6x6 SPD system and solves via LDL^T.

struct SolverParams {
  dt: f32,
  gravity_x: f32,
  gravity_y: f32,
  gravity_z: f32,
  penalty_min: f32,
  penalty_max: f32,
  beta: f32,
  alpha: f32,
  num_bodies: u32,
  num_constraints: u32,
  num_bodies_in_group: u32,
  is_stabilization: u32,
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

@group(0) @binding(0) var<uniform> params: SolverParams;
@group(0) @binding(1) var<storage, read_write> body_state: array<f32>;
@group(0) @binding(2) var<storage, read> body_prev: array<f32>;
@group(0) @binding(3) var<storage, read> constraints: array<ConstraintRow3D>;
@group(0) @binding(4) var<storage, read> color_body_indices: array<u32>;
@group(0) @binding(5) var<storage, read> body_constraint_ranges: array<u32>;
@group(0) @binding(6) var<storage, read> constraint_indices: array<u32>;

// 6x6 LDL^T solver
fn solve_ldl6(A: array<f32, 36>, b: array<f32, 6>) -> array<f32, 6> {
  var L: array<f32, 36>;
  var D: array<f32, 6>;
  // Regularize to prevent f32 catastrophic cancellation when penalty >> mass/dt²
  var max_diag6: f32 = 0.0;
  for (var i = 0u; i < 6u; i++) { max_diag6 = max(max_diag6, A[i * 6u + i]); }
  let eps6 = 1e-6 * max_diag6;

  for (var j = 0u; j < 6u; j++) {
    var sum_d = A[j * 6u + j] + eps6;
    for (var k = 0u; k < j; k++) {
      let ljk = L[k * 6u + j];
      sum_d -= ljk * ljk * D[k];
    }
    D[j] = sum_d;
    if (D[j] <= 0.0) {
      var zero: array<f32, 6>;
      return zero;
    }

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
    for (var k = 0u; k < i; k++) { s -= L[k * 6u + i] * y[k]; }
    y[i] = s;
  }

  var z: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) { z[i] = y[i] / D[i]; }

  var x: array<f32, 6>;
  for (var ii = 0u; ii < 6u; ii++) {
    let i = 5u - ii;
    var s = z[i];
    for (var k = i + 1u; k < 6u; k++) { s -= L[i * 6u + k] * x[k]; }
    x[i] = s;
  }
  return x;
}

// Body state 3D layout (20 floats per body):
// [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, mass, Ix, Iy, Iz, pad, pad, pad]
// Body prev 3D layout (14 floats per body):
// [px, py, pz, pqw, pqx, pqy, pqz, ix, iy, iz, iqw, iqx, iqy, iqz]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let thread_id = gid.x;
  if (thread_id >= params.num_bodies_in_group) { return; }

  let body_idx = color_body_indices[thread_id];
  let dt = params.dt;
  let dt2 = dt * dt;

  let base = body_idx * 20u;
  let mass = body_state[base + 13u];
  if (mass <= 0.0) { return; }

  let Ix = body_state[base + 14u];
  let Iy = body_state[base + 15u];
  let Iz = body_state[base + 16u];

  let x = body_state[base + 0u];
  let y = body_state[base + 1u];
  let z = body_state[base + 2u];

  let prev_base = body_idx * 14u;
  let prev_x = body_prev[prev_base + 0u];
  let prev_y = body_prev[prev_base + 1u];
  let prev_z = body_prev[prev_base + 2u];
  let inertial_x = body_prev[prev_base + 7u];
  let inertial_y = body_prev[prev_base + 8u];
  let inertial_z = body_prev[prev_base + 9u];

  // Read inertial quaternion from body_prev
  let iqw = body_prev[prev_base + 10u];
  let iqx = body_prev[prev_base + 11u];
  let iqy = body_prev[prev_base + 12u];
  let iqz = body_prev[prev_base + 13u];

  // Current quaternion
  let cur_qw = body_state[base + 3u];
  let cur_qx = body_state[base + 4u];
  let cur_qy = body_state[base + 5u];
  let cur_qz = body_state[base + 6u];

  // Quaternion difference: dq = q * conj(q_inertial)
  // Small angle approximation: dtheta ≈ 2 * vec3(dq.xyz)
  let dqx_r = cur_qx * iqw - cur_qw * iqx + cur_qz * iqy - cur_qy * iqz;
  let dqy_r = cur_qy * iqw - cur_qw * iqy + cur_qx * iqz - cur_qz * iqx;
  let dqz_r = cur_qz * iqw - cur_qw * iqz + cur_qy * iqx - cur_qx * iqy;

  // Initialize 6x6 LHS diagonal
  var lhs: array<f32, 36>;
  for (var i = 0u; i < 36u; i++) { lhs[i] = 0.0; }
  lhs[0 * 6u + 0u] = mass / dt2;
  lhs[1 * 6u + 1u] = mass / dt2;
  lhs[2 * 6u + 2u] = mass / dt2;
  lhs[3 * 6u + 3u] = Ix / dt2;
  lhs[4 * 6u + 4u] = Iy / dt2;
  lhs[5 * 6u + 5u] = Iz / dt2;

  var rhs: array<f32, 6>;
  rhs[0] = mass / dt2 * (x - inertial_x);
  rhs[1] = mass / dt2 * (y - inertial_y);
  rhs[2] = mass / dt2 * (z - inertial_z);
  rhs[3] = Ix / dt2 * 2.0 * dqx_r;
  rhs[4] = Iy / dt2 * 2.0 * dqy_r;
  rhs[5] = Iz / dt2 * 2.0 * dqz_r;

  // Accumulate constraint contributions
  let range_base = body_idx * 2u;
  let constraint_start = body_constraint_ranges[range_base + 0u];
  let constraint_count = body_constraint_ranges[range_base + 1u];

  for (var ci = 0u; ci < constraint_count; ci++) {
    let cr_idx = constraint_indices[constraint_start + ci];
    let cr = constraints[cr_idx];
    if (cr.is_active == 0u) { continue; }

    var J: array<f32, 6>;
    if (cr.body_a == i32(body_idx)) {
      J[0] = cr.jacobian_a_lin.x; J[1] = cr.jacobian_a_lin.y; J[2] = cr.jacobian_a_lin.z;
      J[3] = cr.jacobian_a_ang.x; J[4] = cr.jacobian_a_ang.y; J[5] = cr.jacobian_a_ang.z;
    } else {
      J[0] = cr.jacobian_b_lin.x; J[1] = cr.jacobian_b_lin.y; J[2] = cr.jacobian_b_lin.z;
      J[3] = cr.jacobian_b_ang.x; J[4] = cr.jacobian_b_ang.y; J[5] = cr.jacobian_b_ang.z;
    }

    // Full Taylor-series constraint evaluation: C = C0*(1-alpha) + J_A·dp_A + J_B·dp_B
    var c_eval = cr.c0 * (1.0 - params.alpha);
    if (cr.body_a >= 0) {
      let ba = u32(cr.body_a) * 20u;
      let bap = u32(cr.body_a) * 14u;
      let dpx = body_state[ba + 0u] - body_prev[bap + 0u];
      let dpy = body_state[ba + 1u] - body_prev[bap + 1u];
      let dpz = body_state[ba + 2u] - body_prev[bap + 2u];
      c_eval += cr.jacobian_a_lin.x * dpx + cr.jacobian_a_lin.y * dpy + cr.jacobian_a_lin.z * dpz;
      // Angular displacement via quaternion difference: dq = q * conj(q_prev)
      let qw_a = body_state[ba + 3u]; let qx_a = body_state[ba + 4u];
      let qy_a = body_state[ba + 5u]; let qz_a = body_state[ba + 6u];
      let pqw_a = body_prev[bap + 3u]; let pqx_a = body_prev[bap + 4u];
      let pqy_a = body_prev[bap + 5u]; let pqz_a = body_prev[bap + 6u];
      let dqx_a = pqw_a * qx_a - pqx_a * qw_a - pqy_a * qz_a + pqz_a * qy_a;
      let dqy_a = pqw_a * qy_a + pqx_a * qz_a - pqy_a * qw_a - pqz_a * qx_a;
      let dqz_a = pqw_a * qz_a - pqx_a * qy_a + pqy_a * qx_a - pqz_a * qw_a;
      c_eval += cr.jacobian_a_ang.x * 2.0 * dqx_a
             + cr.jacobian_a_ang.y * 2.0 * dqy_a
             + cr.jacobian_a_ang.z * 2.0 * dqz_a;
    }
    if (cr.body_b >= 0) {
      let bb = u32(cr.body_b) * 20u;
      let bbp = u32(cr.body_b) * 14u;
      let dpx = body_state[bb + 0u] - body_prev[bbp + 0u];
      let dpy = body_state[bb + 1u] - body_prev[bbp + 1u];
      let dpz = body_state[bb + 2u] - body_prev[bbp + 2u];
      c_eval += cr.jacobian_b_lin.x * dpx + cr.jacobian_b_lin.y * dpy + cr.jacobian_b_lin.z * dpz;
      let qw_b = body_state[bb + 3u]; let qx_b = body_state[bb + 4u];
      let qy_b = body_state[bb + 5u]; let qz_b = body_state[bb + 6u];
      let pqw_b = body_prev[bbp + 3u]; let pqx_b = body_prev[bbp + 4u];
      let pqy_b = body_prev[bbp + 5u]; let pqz_b = body_prev[bbp + 6u];
      let dqx_b = pqw_b * qx_b - pqx_b * qw_b - pqy_b * qz_b + pqz_b * qy_b;
      let dqy_b = pqw_b * qy_b + pqx_b * qz_b - pqy_b * qw_b - pqz_b * qx_b;
      let dqz_b = pqw_b * qz_b - pqx_b * qy_b + pqy_b * qx_b - pqz_b * qw_b;
      c_eval += cr.jacobian_b_ang.x * 2.0 * dqx_b
             + cr.jacobian_b_ang.y * 2.0 * dqy_b
             + cr.jacobian_b_ang.z * 2.0 * dqz_b;
    }

    // Stiffness guard
    var lambda_for_primal = cr.lambda;
    if (cr.stiffness < 1e30) { lambda_for_primal = 0.0; }

    var f = cr.penalty * c_eval + lambda_for_primal;
    f = clamp(f, cr.fmin, cr.fmax);

    for (var k = 0u; k < 6u; k++) { rhs[k] += J[k] * f; }
    for (var i = 0u; i < 6u; i++) {
      for (var j = 0u; j < 6u; j++) {
        lhs[j * 6u + i] += J[i] * J[j] * cr.penalty;
      }
    }

    // Geometric stiffness (diagonal lumping) for angular DOFs
    let abs_f = abs(f);
    var H_ang: vec3<f32>;
    if (cr.body_a == i32(body_idx)) {
      H_ang = cr.hessian_diag_a_ang.xyz;
    } else {
      H_ang = cr.hessian_diag_b_ang.xyz;
    }
    lhs[3u * 6u + 3u] += abs(H_ang.x) * abs_f;
    lhs[4u * 6u + 4u] += abs(H_ang.y) * abs_f;
    lhs[5u * 6u + 5u] += abs(H_ang.z) * abs_f;
  }

  // Check diagonal validity
  for (var i = 0u; i < 6u; i++) {
    if (lhs[i * 6u + i] <= 0.0) { return; }
  }

  let delta = solve_ldl6(lhs, rhs);

  body_state[base + 0u] = x - delta[0];
  body_state[base + 1u] = y - delta[1];
  body_state[base + 2u] = z - delta[2];

  // Quaternion update from angular correction
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

  let qlen = sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz);
  if (qlen > 0.0) {
    nqw /= qlen; nqx /= qlen; nqy /= qlen; nqz /= qlen;
  }

  body_state[base + 3u] = nqw;
  body_state[base + 4u] = nqx;
  body_state[base + 5u] = nqy;
  body_state[base + 6u] = nqz;
}
`;

export const RADIX_SORT_WGSL = `// ─── GPU Radix Sort (4-bit LSD, 32-bit key-value pairs) ─────────────────────
// Three entry points per digit pass:
//   1. radix_histogram  — count digit frequencies per workgroup tile
//   2. radix_prefix_sum — exclusive prefix sum over global histogram
//   3. radix_scatter    — scatter keys+values to sorted positions
//
// Buffers:
//   keys_in/out, vals_in/out — ping-pong key-value arrays
//   histogram — [16 * num_workgroups] digit counts
//   prefix_sum — [16 * num_workgroups] scanned offsets
//   params — [num_elements, digit_shift]

const RADIX: u32 = 16u;       // 4-bit radix
const WG_SIZE: u32 = 256u;    // threads per workgroup
const TILE_SIZE: u32 = 256u;  // elements per workgroup (1 per thread)

struct SortParams {
  num_elements: u32,
  digit_shift: u32,
  num_workgroups: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> histogram: array<u32>;

var<workgroup> local_hist: array<u32, 256>;  // 16 digits * 16 padding-safe slots? No, just 16 * WG reduction
var<workgroup> shared_keys: array<u32, 256>;

// ─── Pass 1: Histogram ──────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn radix_histogram(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  // Clear local histogram (16 bins, each thread clears one if tid < 16)
  if (lid.x < RADIX) {
    local_hist[lid.x] = 0u;
  }
  workgroupBarrier();

  // Each thread processes one element
  let idx = gid.x;
  if (idx < params.num_elements) {
    let key = keys_in[idx];
    let digit = (key >> params.digit_shift) & 0xFu;
    // Atomic increment in workgroup shared memory
    atomicAdd(&local_hist[digit], 1u);
  }
  workgroupBarrier();

  // Write local histogram to global histogram
  // Layout: histogram[digit * num_workgroups + workgroup_id]
  if (lid.x < RADIX) {
    histogram[lid.x * params.num_workgroups + wid.x] = atomicLoad(&local_hist[lid.x]);
  }
}

// ─── Pass 2: Prefix Sum (single workgroup scans entire histogram) ───────────
// Handles up to 256 * 16 = 4096 entries (256 workgroups * 16 digits)
// For larger counts, we'd need a multi-level scan, but 256 WGs covers 65536 elements

var<workgroup> scan_temp: array<u32, 4096>;

@compute @workgroup_size(256)
fn radix_prefix_sum(
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let total_bins = RADIX * params.num_workgroups;

  // Load histogram into shared memory (each thread loads multiple elements)
  let items_per_thread = (total_bins + WG_SIZE - 1u) / WG_SIZE;
  for (var i = 0u; i < items_per_thread; i++) {
    let idx = lid.x + i * WG_SIZE;
    if (idx < total_bins) {
      scan_temp[idx] = histogram[idx];
    }
  }
  workgroupBarrier();

  // Blelloch exclusive prefix sum (up-sweep then down-sweep)
  // Up-sweep (reduce)
  var offset = 1u;
  var n = total_bins;
  // Round n up to next power of 2 for Blelloch scan
  var n_pow2 = 1u;
  while (n_pow2 < n) {
    n_pow2 = n_pow2 << 1u;
  }

  // Up-sweep
  var d = n_pow2 >> 1u;
  while (d > 0u) {
    workgroupBarrier();
    let items = (n_pow2 / (offset * 2u) + WG_SIZE - 1u) / WG_SIZE;
    for (var i = 0u; i < items; i++) {
      let t = lid.x + i * WG_SIZE;
      let ai = offset * (2u * t + 1u) - 1u;
      let bi = offset * (2u * t + 2u) - 1u;
      if (bi < n_pow2 && ai < n_pow2) {
        scan_temp[bi] = scan_temp[bi] + scan_temp[ai];
      }
    }
    offset = offset * 2u;
    d = d >> 1u;
  }

  workgroupBarrier();
  // Set last element to 0 for exclusive scan
  if (lid.x == 0u) {
    scan_temp[n_pow2 - 1u] = 0u;
  }

  // Down-sweep
  d = 1u;
  while (d < n_pow2) {
    offset = offset >> 1u;
    workgroupBarrier();
    let items = (n_pow2 / (d * 2u) + WG_SIZE - 1u) / WG_SIZE;
    for (var i = 0u; i < items; i++) {
      let t = lid.x + i * WG_SIZE;
      let ai = offset * (2u * t + 1u) - 1u;
      let bi = offset * (2u * t + 2u) - 1u;
      if (bi < n_pow2 && ai < n_pow2) {
        let temp = scan_temp[ai];
        scan_temp[ai] = scan_temp[bi];
        scan_temp[bi] = scan_temp[bi] + temp;
      }
    }
    d = d * 2u;
  }
  workgroupBarrier();

  // Write back to global histogram (now contains prefix sums)
  for (var i = 0u; i < items_per_thread; i++) {
    let idx = lid.x + i * WG_SIZE;
    if (idx < total_bins) {
      histogram[idx] = scan_temp[idx];
    }
  }
}

// ─── Pass 3: Scatter ────────────────────────────────────────────────────────

var<workgroup> local_offsets: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn radix_scatter(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  // Load this workgroup's prefix sum offsets
  if (lid.x < RADIX) {
    atomicStore(&local_offsets[lid.x], histogram[lid.x * params.num_workgroups + wid.x]);
  }
  workgroupBarrier();

  let idx = gid.x;
  if (idx < params.num_elements) {
    let key = keys_in[idx];
    let val = vals_in[idx];
    let digit = (key >> params.digit_shift) & 0xFu;

    // Get destination index via atomic increment
    let dest = atomicAdd(&local_offsets[digit], 1u);
    keys_out[dest] = key;
    vals_out[dest] = val;
  }
}
`;

