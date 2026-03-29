// Bottom-up AABB refit for 2D BVH.
// One thread per leaf. Each leaf writes its AABB, then walks up the parent
// chain using atomic visit counts to ensure each internal node is processed
// exactly once (by the second visitor).
//
// node_aabb uses atomic<u32> with bitcast<f32/u32> to ensure cross-thread
// visibility of AABB writes within the same compute pass. This is required
// for correctness on software GPU backends (SwiftShader) where non-atomic
// storage writes may not be flushed before/after atomic counter operations.

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

// Output: node AABBs for all 2N-1 nodes, 4 u32 (bitcast f32) each
// Layout: [internal_0, internal_1, ..., internal_{N-2}, leaf_0, leaf_1, ..., leaf_{N-1}]
// Uses atomic<u32> to ensure cross-thread visibility during bottom-up refit.
@group(0) @binding(6) var<storage, read_write> node_aabb: array<atomic<u32>>;

// Atomic visit counters for internal nodes (size N-1), initialised to 0
@group(0) @binding(7) var<storage, read_write> visit_count: array<atomic<u32>>;

fn store_aabb(node: u32, min_x: f32, min_y: f32, max_x: f32, max_y: f32) {
  let base = node * 4u;
  atomicStore(&node_aabb[base + 0u], bitcast<u32>(min_x));
  atomicStore(&node_aabb[base + 1u], bitcast<u32>(min_y));
  atomicStore(&node_aabb[base + 2u], bitcast<u32>(max_x));
  atomicStore(&node_aabb[base + 3u], bitcast<u32>(max_y));
}

fn load_aabb_min_x(node: u32) -> f32 { return bitcast<f32>(atomicLoad(&node_aabb[node * 4u + 0u])); }
fn load_aabb_min_y(node: u32) -> f32 { return bitcast<f32>(atomicLoad(&node_aabb[node * 4u + 1u])); }
fn load_aabb_max_x(node: u32) -> f32 { return bitcast<f32>(atomicLoad(&node_aabb[node * 4u + 2u])); }
fn load_aabb_max_y(node: u32) -> f32 { return bitcast<f32>(atomicLoad(&node_aabb[node * 4u + 3u])); }

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

  // Step 1: Write leaf AABB using atomic stores
  let body_idx = sorted_indices[leaf_idx];
  let src_base = body_idx * 4u;
  let dst_node = ni + leaf_idx;

  store_aabb(dst_node,
    aabb_buffer[src_base + 0u],
    aabb_buffer[src_base + 1u],
    aabb_buffer[src_base + 2u],
    aabb_buffer[src_base + 3u]);

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

    let union_min_x = min(load_aabb_min_x(ln), load_aabb_min_x(rn));
    let union_min_y = min(load_aabb_min_y(ln), load_aabb_min_y(rn));
    let union_max_x = max(load_aabb_max_x(ln), load_aabb_max_x(rn));
    let union_max_y = max(load_aabb_max_y(ln), load_aabb_max_y(rn));

    store_aabb(pu, union_min_x, union_min_y, union_max_x, union_max_y);

    // Move to parent's parent
    p = parent_buf[pu];
  }
}
