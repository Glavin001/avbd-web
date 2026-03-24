// Bottom-up AABB refit for 3D BVH.
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
