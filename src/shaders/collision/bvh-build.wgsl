// Karras 2012 LBVH hierarchy construction.
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
