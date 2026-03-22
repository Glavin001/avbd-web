import { describe, it, expect } from 'vitest';
import {
  vec2, vec2Add, vec2Sub, vec2Scale, vec2Dot, vec2Cross, vec2Length,
  vec2Normalize, vec2Neg, vec2Perp, vec2Rotate,
  vec3, vec3Add, vec3Sub, vec3Scale, vec3Dot, vec3Cross, vec3Length, vec3Normalize,
  quatIdentity, quatMul, quatNormalize, quatRotateVec3, quatFromAxisAngle,
  mat3Zero, mat3Identity, mat3Scale, mat3Get, mat3Set, mat3Add, mat3MulVec3,
  mat3OuterProduct, solveLDL3, diagonalGeometricStiffness3,
  aabb2DOverlap, aabb2DExpand,
} from '../src/core/math.js';

describe('Vec2 operations', () => {
  it('should create a vec2', () => {
    const v = vec2(3, 4);
    expect(v.x).toBe(3);
    expect(v.y).toBe(4);
  });

  it('should add two vec2s', () => {
    const r = vec2Add(vec2(1, 2), vec2(3, 4));
    expect(r.x).toBe(4);
    expect(r.y).toBe(6);
  });

  it('should subtract two vec2s', () => {
    const r = vec2Sub(vec2(5, 7), vec2(2, 3));
    expect(r.x).toBe(3);
    expect(r.y).toBe(4);
  });

  it('should scale a vec2', () => {
    const r = vec2Scale(vec2(2, 3), 4);
    expect(r.x).toBe(8);
    expect(r.y).toBe(12);
  });

  it('should compute dot product', () => {
    expect(vec2Dot(vec2(1, 2), vec2(3, 4))).toBe(11);
  });

  it('should compute cross product (scalar in 2D)', () => {
    expect(vec2Cross(vec2(1, 0), vec2(0, 1))).toBe(1);
    expect(vec2Cross(vec2(0, 1), vec2(1, 0))).toBe(-1);
  });

  it('should compute length', () => {
    expect(vec2Length(vec2(3, 4))).toBe(5);
    expect(vec2Length(vec2(0, 0))).toBe(0);
  });

  it('should normalize', () => {
    const r = vec2Normalize(vec2(3, 4));
    expect(r.x).toBeCloseTo(0.6);
    expect(r.y).toBeCloseTo(0.8);
  });

  it('should handle zero vector normalization', () => {
    const r = vec2Normalize(vec2(0, 0));
    expect(r.x).toBe(0);
    expect(r.y).toBe(0);
  });

  it('should negate', () => {
    const r = vec2Neg(vec2(3, -4));
    expect(r.x).toBe(-3);
    expect(r.y).toBe(4);
  });

  it('should compute perpendicular', () => {
    const r = vec2Perp(vec2(1, 0));
    expect(r.x).toBeCloseTo(0);
    expect(r.y).toBe(1);
  });

  it('should rotate by 90 degrees', () => {
    const r = vec2Rotate(vec2(1, 0), Math.PI / 2);
    expect(r.x).toBeCloseTo(0, 5);
    expect(r.y).toBeCloseTo(1, 5);
  });

  it('should rotate by 180 degrees', () => {
    const r = vec2Rotate(vec2(1, 0), Math.PI);
    expect(r.x).toBeCloseTo(-1, 5);
    expect(r.y).toBeCloseTo(0, 5);
  });
});

describe('Vec3 operations', () => {
  it('should create, add, sub, scale', () => {
    const a = vec3(1, 2, 3);
    const b = vec3(4, 5, 6);
    const sum = vec3Add(a, b);
    expect(sum).toEqual({ x: 5, y: 7, z: 9 });
    const diff = vec3Sub(b, a);
    expect(diff).toEqual({ x: 3, y: 3, z: 3 });
    const scaled = vec3Scale(a, 2);
    expect(scaled).toEqual({ x: 2, y: 4, z: 6 });
  });

  it('should compute dot product', () => {
    expect(vec3Dot(vec3(1, 2, 3), vec3(4, 5, 6))).toBe(32);
  });

  it('should compute cross product', () => {
    const r = vec3Cross(vec3(1, 0, 0), vec3(0, 1, 0));
    expect(r).toEqual({ x: 0, y: 0, z: 1 });
  });

  it('should compute length and normalize', () => {
    expect(vec3Length(vec3(3, 4, 0))).toBe(5);
    const n = vec3Normalize(vec3(0, 0, 5));
    expect(n).toEqual({ x: 0, y: 0, z: 1 });
  });
});

describe('Quaternion operations', () => {
  it('should create identity quaternion', () => {
    const q = quatIdentity();
    expect(q).toEqual({ w: 1, x: 0, y: 0, z: 0 });
  });

  it('should multiply identity by identity', () => {
    const q = quatMul(quatIdentity(), quatIdentity());
    expect(q.w).toBeCloseTo(1);
    expect(q.x).toBeCloseTo(0);
    expect(q.y).toBeCloseTo(0);
    expect(q.z).toBeCloseTo(0);
  });

  it('should rotate vector by identity quaternion', () => {
    const v = vec3(1, 2, 3);
    const r = quatRotateVec3(quatIdentity(), v);
    expect(r.x).toBeCloseTo(1);
    expect(r.y).toBeCloseTo(2);
    expect(r.z).toBeCloseTo(3);
  });

  it('should rotate (1,0,0) by 90 degrees around Z axis', () => {
    const q = quatFromAxisAngle(vec3(0, 0, 1), Math.PI / 2);
    const r = quatRotateVec3(q, vec3(1, 0, 0));
    expect(r.x).toBeCloseTo(0, 5);
    expect(r.y).toBeCloseTo(1, 5);
    expect(r.z).toBeCloseTo(0, 5);
  });

  it('should normalize quaternion', () => {
    const q = quatNormalize({ w: 2, x: 0, y: 0, z: 0 });
    expect(q.w).toBeCloseTo(1);
  });

  it('should compose rotations via multiplication', () => {
    const q90z = quatFromAxisAngle(vec3(0, 0, 1), Math.PI / 2);
    const q180z = quatMul(q90z, q90z);
    const r = quatRotateVec3(q180z, vec3(1, 0, 0));
    expect(r.x).toBeCloseTo(-1, 5);
    expect(r.y).toBeCloseTo(0, 5);
    expect(r.z).toBeCloseTo(0, 5);
  });
});

describe('Mat3 operations', () => {
  it('should create zero and identity matrices', () => {
    const z = mat3Zero();
    expect(z[0]).toBe(0);
    const id = mat3Identity();
    expect(mat3Get(id, 0, 0)).toBe(1);
    expect(mat3Get(id, 1, 1)).toBe(1);
    expect(mat3Get(id, 2, 2)).toBe(1);
    expect(mat3Get(id, 0, 1)).toBe(0);
  });

  it('should get and set elements', () => {
    const m = mat3Zero();
    mat3Set(m, 1, 2, 42);
    expect(mat3Get(m, 1, 2)).toBe(42);
    expect(mat3Get(m, 2, 1)).toBe(0);
  });

  it('should scale matrix', () => {
    const m = mat3Identity();
    const s = mat3Scale(m, 3);
    expect(mat3Get(s, 0, 0)).toBe(3);
    expect(mat3Get(s, 1, 1)).toBe(3);
  });

  it('should add matrices', () => {
    const a = mat3Identity();
    const b = mat3Identity();
    const sum = mat3Add(a, b);
    expect(mat3Get(sum, 0, 0)).toBe(2);
    expect(mat3Get(sum, 0, 1)).toBe(0);
  });

  it('should multiply matrix by vector', () => {
    const m = mat3Identity();
    const r = mat3MulVec3(m, [3, 4, 5]);
    expect(r).toEqual([3, 4, 5]);
  });

  it('should compute outer product', () => {
    const v: [number, number, number] = [1, 2, 3];
    const m = mat3OuterProduct(v);
    expect(mat3Get(m, 0, 0)).toBe(1);  // 1*1
    expect(mat3Get(m, 0, 1)).toBe(2);  // 1*2
    expect(mat3Get(m, 1, 0)).toBe(2);  // 2*1
    expect(mat3Get(m, 1, 1)).toBe(4);  // 2*2
    expect(mat3Get(m, 2, 2)).toBe(9);  // 3*3
  });
});

describe('LDL3 solver', () => {
  it('should solve identity system', () => {
    const A = mat3Identity();
    const b: [number, number, number] = [1, 2, 3];
    const x = solveLDL3(A, b);
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(2);
    expect(x[2]).toBeCloseTo(3);
  });

  it('should solve diagonal system', () => {
    const A = mat3Zero();
    mat3Set(A, 0, 0, 2);
    mat3Set(A, 1, 1, 4);
    mat3Set(A, 2, 2, 8);
    const x = solveLDL3(A, [6, 12, 24]);
    expect(x[0]).toBeCloseTo(3);
    expect(x[1]).toBeCloseTo(3);
    expect(x[2]).toBeCloseTo(3);
  });

  it('should solve SPD system', () => {
    // A = [[4, 2, 1], [2, 5, 2], [1, 2, 6]]  — symmetric positive definite
    const A = mat3Zero();
    mat3Set(A, 0, 0, 4); mat3Set(A, 0, 1, 2); mat3Set(A, 0, 2, 1);
    mat3Set(A, 1, 0, 2); mat3Set(A, 1, 1, 5); mat3Set(A, 1, 2, 2);
    mat3Set(A, 2, 0, 1); mat3Set(A, 2, 1, 2); mat3Set(A, 2, 2, 6);
    const b: [number, number, number] = [1, 2, 3];
    const x = solveLDL3(A, b);

    // Verify: A * x ≈ b
    const Ax = mat3MulVec3(A, x);
    expect(Ax[0]).toBeCloseTo(1, 8);
    expect(Ax[1]).toBeCloseTo(2, 8);
    expect(Ax[2]).toBeCloseTo(3, 8);
  });

  it('should solve mass-inertia-like system', () => {
    // Typical AVBD system: mass matrix + penalty Jacobian contributions
    const A = mat3Zero();
    mat3Set(A, 0, 0, 1000);  // mass/dt^2
    mat3Set(A, 1, 1, 1000);
    mat3Set(A, 2, 2, 500);   // inertia/dt^2
    // Add off-diagonal coupling from Jacobian
    mat3Set(A, 0, 2, 10); mat3Set(A, 2, 0, 10);
    mat3Set(A, 1, 2, 20); mat3Set(A, 2, 1, 20);

    const b: [number, number, number] = [100, 200, 50];
    const x = solveLDL3(A, b);
    const Ax = mat3MulVec3(A, x);
    expect(Ax[0]).toBeCloseTo(100, 4);
    expect(Ax[1]).toBeCloseTo(200, 4);
    expect(Ax[2]).toBeCloseTo(50, 4);
  });
});

describe('Geometric stiffness', () => {
  it('should compute diagonal geometric stiffness', () => {
    const m = diagonalGeometricStiffness3([2, 3, 4], 5);
    expect(mat3Get(m, 0, 0)).toBe(10);  // |2| * 5
    expect(mat3Get(m, 1, 1)).toBe(15);  // |3| * 5
    expect(mat3Get(m, 2, 2)).toBe(20);  // |4| * 5
    expect(mat3Get(m, 0, 1)).toBe(0);   // off-diagonal = 0
  });

  it('should handle negative Hessian values', () => {
    const m = diagonalGeometricStiffness3([-2, -3, -4], 5);
    expect(mat3Get(m, 0, 0)).toBe(10);
    expect(mat3Get(m, 1, 1)).toBe(15);
    expect(mat3Get(m, 2, 2)).toBe(20);
  });
});

describe('AABB2D', () => {
  it('should detect overlapping AABBs', () => {
    const a = { minX: 0, minY: 0, maxX: 2, maxY: 2 };
    const b = { minX: 1, minY: 1, maxX: 3, maxY: 3 };
    expect(aabb2DOverlap(a, b)).toBe(true);
  });

  it('should detect non-overlapping AABBs', () => {
    const a = { minX: 0, minY: 0, maxX: 1, maxY: 1 };
    const b = { minX: 2, minY: 2, maxX: 3, maxY: 3 };
    expect(aabb2DOverlap(a, b)).toBe(false);
  });

  it('should detect touching AABBs as overlapping', () => {
    const a = { minX: 0, minY: 0, maxX: 1, maxY: 1 };
    const b = { minX: 1, minY: 0, maxX: 2, maxY: 1 };
    expect(aabb2DOverlap(a, b)).toBe(true);
  });

  it('should expand AABB by margin', () => {
    const a = { minX: 0, minY: 0, maxX: 1, maxY: 1 };
    const expanded = aabb2DExpand(a, 0.5);
    expect(expanded.minX).toBe(-0.5);
    expect(expanded.minY).toBe(-0.5);
    expect(expanded.maxX).toBe(1.5);
    expect(expanded.maxY).toBe(1.5);
  });
});
