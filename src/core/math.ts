/**
 * CPU-side math utilities for the AVBD physics engine.
 * Mirrors the math operations from avbd-demo2d/source/maths.h.
 */

import type { Vec2, Vec3, Quat } from './types.js';

// ─── Vec2 Operations ────────────────────────────────────────────────────────

export function vec2(x: number, y: number): Vec2 {
  return { x, y };
}

export function vec2Add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function vec2Sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function vec2Scale(v: Vec2, s: number): Vec2 {
  return { x: v.x * s, y: v.y * s };
}

export function vec2Dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

export function vec2Cross(a: Vec2, b: Vec2): number {
  return a.x * b.y - a.y * b.x;
}

export function vec2Length(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

export function vec2Normalize(v: Vec2): Vec2 {
  const len = vec2Length(v);
  if (len < 1e-10) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
}

export function vec2Neg(v: Vec2): Vec2 {
  return { x: -v.x, y: -v.y };
}

export function vec2Perp(v: Vec2): Vec2 {
  return { x: -v.y, y: v.x };
}

/**
 * Rotate a 2D vector by angle (radians).
 */
export function vec2Rotate(v: Vec2, angle: number): Vec2 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return { x: c * v.x - s * v.y, y: s * v.x + c * v.y };
}

// ─── Vec3 Operations ────────────────────────────────────────────────────────

export function vec3(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function vec3Scale(v: Vec3, s: number): Vec3 {
  return { x: v.x * s, y: v.y * s, z: v.z * s };
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function vec3Length(v: Vec3): number {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

export function vec3Normalize(v: Vec3): Vec3 {
  const len = vec3Length(v);
  if (len < 1e-10) return { x: 0, y: 0, z: 0 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

// ─── Quaternion Operations ──────────────────────────────────────────────────

export function quatIdentity(): Quat {
  return { w: 1, x: 0, y: 0, z: 0 };
}

export function quatMul(a: Quat, b: Quat): Quat {
  return {
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

export function quatNormalize(q: Quat): Quat {
  const len = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (len < 1e-10) return quatIdentity();
  return { w: q.w / len, x: q.x / len, y: q.y / len, z: q.z / len };
}

export function quatRotateVec3(q: Quat, v: Vec3): Vec3 {
  // q * v * q^-1 (optimized)
  const qv = { x: q.x, y: q.y, z: q.z };
  const uv = vec3Cross(qv, v);
  const uuv = vec3Cross(qv, uv);
  return vec3Add(v, vec3Add(vec3Scale(uv, 2 * q.w), vec3Scale(uuv, 2)));
}

export function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const ha = angle * 0.5;
  const s = Math.sin(ha);
  return { w: Math.cos(ha), x: axis.x * s, y: axis.y * s, z: axis.z * s };
}

// ─── 3x3 Matrix Operations (for 2D solver) ─────────────────────────────────

/**
 * 3x3 matrix stored as flat array in column-major order.
 * [m00, m10, m20, m01, m11, m21, m02, m12, m22]
 */
export type Mat3 = Float64Array;

export function mat3Zero(): Mat3 {
  return new Float64Array(9);
}

export function mat3Identity(): Mat3 {
  const m = new Float64Array(9);
  m[0] = 1; m[4] = 1; m[8] = 1;
  return m;
}

export function mat3Scale(m: Mat3, s: number): Mat3 {
  const r = new Float64Array(9);
  for (let i = 0; i < 9; i++) r[i] = m[i] * s;
  return r;
}

/** Get element at row i, col j (column-major) */
export function mat3Get(m: Mat3, i: number, j: number): number {
  return m[j * 3 + i];
}

/** Set element at row i, col j (column-major) */
export function mat3Set(m: Mat3, i: number, j: number, v: number): void {
  m[j * 3 + i] = v;
}

/** Add two 3x3 matrices */
export function mat3Add(a: Mat3, b: Mat3): Mat3 {
  const r = new Float64Array(9);
  for (let i = 0; i < 9; i++) r[i] = a[i] + b[i];
  return r;
}

/** Multiply 3x3 matrix by 3-vector: result = M * v */
export function mat3MulVec3(m: Mat3, v: [number, number, number]): [number, number, number] {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

/**
 * Outer product: result = v * v^T (produces a 3x3 symmetric matrix)
 */
export function mat3OuterProduct(v: [number, number, number]): Mat3 {
  const m = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      m[j * 3 + i] = v[i] * v[j];
    }
  }
  return m;
}

/**
 * LDL^T factorization and solve for a 3x3 SPD matrix.
 * Solves A * x = b in-place.
 * Based on avbd-demo2d/source/maths.h LDL implementation.
 */
export function solveLDL3(A: Mat3, b: [number, number, number]): [number, number, number] {
  // Copy A to avoid mutation
  const L = new Float64Array(9);
  const D = new Float64Array(3);

  // LDL^T decomposition (column-major)
  // Column 0
  D[0] = mat3Get(A, 0, 0);
  L[0] = 1;
  L[1] = mat3Get(A, 1, 0) / D[0];
  L[2] = mat3Get(A, 2, 0) / D[0];

  // Column 1
  D[1] = mat3Get(A, 1, 1) - L[1] * L[1] * D[0];
  L[3] = 0;
  L[4] = 1;
  L[5] = (mat3Get(A, 2, 1) - L[2] * L[1] * D[0]) / D[1];

  // Column 2
  D[2] = mat3Get(A, 2, 2) - L[2] * L[2] * D[0] - L[5] * L[5] * D[1];
  L[6] = 0;
  L[7] = 0;
  L[8] = 1;

  // Forward substitution: L * y = b
  const y: [number, number, number] = [0, 0, 0];
  y[0] = b[0];
  y[1] = b[1] - L[1] * y[0];
  y[2] = b[2] - L[2] * y[0] - L[5] * y[1];

  // Diagonal solve: D * z = y
  const z: [number, number, number] = [y[0] / D[0], y[1] / D[1], y[2] / D[2]];

  // Back substitution: L^T * x = z
  const x: [number, number, number] = [0, 0, 0];
  x[2] = z[2];
  x[1] = z[1] - L[5] * x[2];
  x[0] = z[0] - L[1] * x[1] - L[2] * x[2];

  return x;
}

/**
 * Compute diagonal geometric stiffness from a 3x3 Hessian matrix.
 * From avbd-demo2d: diag(|H.col(0)|, |H.col(1)|, |H.col(2)|) * |f|
 * Uses column norms of the Hessian for diagonal lumping.
 */
export function diagonalGeometricStiffness3(hessianDiag: [number, number, number], absForce: number): Mat3 {
  const m = mat3Zero();
  m[0] = Math.abs(hessianDiag[0]) * absForce;
  m[4] = Math.abs(hessianDiag[1]) * absForce;
  m[8] = Math.abs(hessianDiag[2]) * absForce;
  return m;
}

// ─── AABB for broadphase ────────────────────────────────────────────────────

export interface AABB2D {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function aabb2DOverlap(a: AABB2D, b: AABB2D): boolean {
  return a.minX <= b.maxX && a.maxX >= b.minX &&
         a.minY <= b.maxY && a.maxY >= b.minY;
}

export function aabb2DExpand(aabb: AABB2D, margin: number): AABB2D {
  return {
    minX: aabb.minX - margin,
    minY: aabb.minY - margin,
    maxX: aabb.maxX + margin,
    maxY: aabb.maxY + margin,
  };
}
