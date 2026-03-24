// ─── GPU Radix Sort (4-bit LSD, 32-bit key-value pairs) ─────────────────────
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
