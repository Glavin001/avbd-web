// ─── AVBD Math Utilities for WGSL ───────────────────────────────────────────
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
