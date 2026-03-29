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

// Output: node AABBs for all 2N-1 nodes, 6 u32 (bitcast f32) each
// Layout: [internal_0, ..., internal_{N-2}, leaf_0, ..., leaf_{N-1}]
// Uses atomic<u32> to ensure cross-thread visibility during bottom-up refit.
@group(0) @binding(6) var<storage, read_write> node_aabb: array<atomic<u32>>;

// Atomic visit counters for internal nodes (size N-1), initialised to 0
@group(0) @binding(7) var<storage, read_write> visit_count: array<atomic<u32>>;

const STRIDE: u32 = 6u;

fn store_aabb(node: u32, v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32) {
  let base = node * STRIDE;
  atomicStore(&node_aabb[base + 0u], bitcast<u32>(v0));
  atomicStore(&node_aabb[base + 1u], bitcast<u32>(v1));
  atomicStore(&node_aabb[base + 2u], bitcast<u32>(v2));
  atomicStore(&node_aabb[base + 3u], bitcast<u32>(v3));
  atomicStore(&node_aabb[base + 4u], bitcast<u32>(v4));
  atomicStore(&node_aabb[base + 5u], bitcast<u32>(v5));
}

fn load_node_aabb(node: u32, offset: u32) -> f32 {
  return bitcast<f32>(atomicLoad(&node_aabb[node * STRIDE + offset]));
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

  // Step 1: Write leaf AABB using atomic stores
  let body_idx = sorted_indices[leaf_idx];
  let src_base = body_idx * STRIDE;
  let dst_node = ni + leaf_idx;

  store_aabb(dst_node,
    aabb_buffer[src_base + 0u],
    aabb_buffer[src_base + 1u],
    aabb_buffer[src_base + 2u],
    aabb_buffer[src_base + 3u],
    aabb_buffer[src_base + 4u],
    aabb_buffer[src_base + 5u]);

  // Step 2: Walk up parent chain
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

    let union_min_x = min(load_node_aabb(ln, 0u), load_node_aabb(rn, 0u));
    let union_min_y = min(load_node_aabb(ln, 1u), load_node_aabb(rn, 1u));
    let union_min_z = min(load_node_aabb(ln, 2u), load_node_aabb(rn, 2u));
    let union_max_x = max(load_node_aabb(ln, 3u), load_node_aabb(rn, 3u));
    let union_max_y = max(load_node_aabb(ln, 4u), load_node_aabb(rn, 4u));
    let union_max_z = max(load_node_aabb(ln, 5u), load_node_aabb(rn, 5u));

    store_aabb(pu, union_min_x, union_min_y, union_min_z, union_max_x, union_max_y, union_max_z);

    // Move to parent's parent
    p = parent_buf[pu];
  }
}
