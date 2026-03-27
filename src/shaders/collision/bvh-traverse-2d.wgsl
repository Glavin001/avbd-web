// BVH traversal for 2D broad-phase pair finding.
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
