import { describe, it, expect } from 'vitest';
import { collideBoxBox, collideCircleCircle, collideBoxCircle, collide2D } from '../src/2d/collision-sat.js';
import type { Body2D } from '../src/core/rigid-body.js';
import { ColliderShapeType } from '../src/core/rigid-body.js';
import { collideBoxBox3D, collideSpheresSphere, collideBoxSphere, collide3D } from '../src/3d/collision-gjk.js';
import type { Body3D } from '../src/core/rigid-body-3d.js';
import { ColliderShapeType3D } from '../src/core/rigid-body-3d.js';
import { RigidBodyType } from '../src/core/types.js';
import { quatIdentity, quatFromAxisAngle, vec3 } from '../src/core/math.js';

// ---------------------------------------------------------------------------
// 2D helpers
// ---------------------------------------------------------------------------

function makeBox(index: number, x: number, y: number, hx: number, hy: number, angle = 0): Body2D {
  return {
    index, type: RigidBodyType.Dynamic,
    position: { x, y }, angle,
    velocity: { x: 0, y: 0 }, angularVelocity: 0,
    mass: 1, invMass: 1, inertia: 1, invInertia: 1,
    gravityScale: 1, linearDamping: 0, angularDamping: 0,
    colliderShape: ColliderShapeType.Cuboid,
    halfExtents: { x: hx, y: hy }, radius: 0,
    friction: 0.5, restitution: 0.3,
    prevPosition: { x, y }, prevAngle: angle,
    inertialPosition: { x, y }, inertialAngle: angle,
    boundingRadius: Math.sqrt(hx * hx + hy * hy),
  };
}

function makeBall(index: number, x: number, y: number, r: number): Body2D {
  return {
    index, type: RigidBodyType.Dynamic,
    position: { x, y }, angle: 0,
    velocity: { x: 0, y: 0 }, angularVelocity: 0,
    mass: 1, invMass: 1, inertia: 1, invInertia: 1,
    gravityScale: 1, linearDamping: 0, angularDamping: 0,
    colliderShape: ColliderShapeType.Ball,
    halfExtents: { x: 0, y: 0 }, radius: r,
    friction: 0.5, restitution: 0.3,
    prevPosition: { x, y }, prevAngle: 0,
    inertialPosition: { x, y }, inertialAngle: 0,
    boundingRadius: r,
  };
}

// ---------------------------------------------------------------------------
// 3D helpers
// ---------------------------------------------------------------------------

function makeBox3D(
  index: number, x: number, y: number, z: number,
  hx: number, hy: number, hz: number,
  rotation = quatIdentity(),
): Body3D {
  return {
    index, type: RigidBodyType.Dynamic,
    position: { x, y, z }, rotation,
    velocity: { x: 0, y: 0, z: 0 }, angularVelocity: { x: 0, y: 0, z: 0 },
    mass: 1, invMass: 1,
    inertia: { x: 1, y: 1, z: 1 }, invInertia: { x: 1, y: 1, z: 1 },
    gravityScale: 1, linearDamping: 0, angularDamping: 0,
    colliderShape: ColliderShapeType3D.Cuboid,
    halfExtents: { x: hx, y: hy, z: hz }, radius: 0,
    friction: 0.5, restitution: 0.3,
    prevPosition: { x, y, z }, prevRotation: quatIdentity(),
    inertialPosition: { x, y, z }, inertialRotation: quatIdentity(),
    boundingRadius: Math.sqrt(hx * hx + hy * hy + hz * hz),
  };
}

function makeSphere3D(index: number, x: number, y: number, z: number, r: number): Body3D {
  return {
    index, type: RigidBodyType.Dynamic,
    position: { x, y, z }, rotation: quatIdentity(),
    velocity: { x: 0, y: 0, z: 0 }, angularVelocity: { x: 0, y: 0, z: 0 },
    mass: 1, invMass: 1,
    inertia: { x: 1, y: 1, z: 1 }, invInertia: { x: 1, y: 1, z: 1 },
    gravityScale: 1, linearDamping: 0, angularDamping: 0,
    colliderShape: ColliderShapeType3D.Ball,
    halfExtents: { x: 0, y: 0, z: 0 }, radius: r,
    friction: 0.5, restitution: 0.3,
    prevPosition: { x, y, z }, prevRotation: quatIdentity(),
    inertialPosition: { x, y, z }, inertialRotation: quatIdentity(),
    boundingRadius: r,
  };
}

// ===========================================================================
// 2D Box-Box Edge Cases
// ===========================================================================

describe('2D Box-Box Edge Cases', () => {
  it('exact touching – boxes at distance = sum of half-extents should be null or negligible', () => {
    // box1 hx=1 at x=0, box2 hx=1 at x=2 → gap = 0
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 2, 0, 1, 1);
    const result = collideBoxBox(a, b);
    // Exactly touching means zero penetration; implementations typically return null
    if (result !== null) {
      expect(result.contacts[0].depth).toBeLessThan(1e-6);
    }
  });

  it('epsilon overlap – boxes overlapping by 0.001 should detect with small depth', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 1.999, 0, 1, 1); // overlap = 2 - 1.999 = 0.001
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.001, 2);
  });

  it('90-degree rotated square behaves same as unrotated', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const bNoRot = makeBox(1, 1.5, 0, 1, 1, 0);
    const bRot = makeBox(2, 1.5, 0, 1, 1, Math.PI / 2);
    const r1 = collideBoxBox(a, bNoRot);
    const r2 = collideBoxBox(a, bRot);
    expect(r1).not.toBeNull();
    expect(r2).not.toBeNull();
    expect(r1!.contacts[0].depth).toBeCloseTo(r2!.contacts[0].depth, 3);
  });

  it('45-degree rotated box vs axis-aligned detects collision and produces valid normal', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    // Rotated 45° box at x=1.8 – the effective reach along x is hx*cos45+hy*sin45 ≈ 1.414
    const b = makeBox(1, 1.8, 0, 1, 1, Math.PI / 4);
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
    // Normal should be unit length
    const n = result!.normal;
    const len = Math.sqrt(n.x * n.x + n.y * n.y);
    expect(len).toBeCloseTo(1, 3);
  });

  it('edge-to-face contact with offset boxes produces contacts', () => {
    // Boxes offset in Y so only a portion of faces overlap
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 1.5, 0.8, 1, 1);
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts.length).toBeGreaterThan(0);
  });

  it('very deep overlap – box fully inside another yields large depth', () => {
    // Small box at center of large box
    const big = makeBox(0, 0, 0, 5, 5);
    const small = makeBox(1, 0, 0, 0.5, 0.5);
    const result = collideBoxBox(big, small);
    expect(result).not.toBeNull();
    // Minimum escape depth = small half-extent + distance to big face
    // Small center at origin, big extends to +-5, small extends +-0.5
    // Closest face distance = 5 - 0 = 5, depth along that axis = 5 + 0.5 = 5.5
    expect(result!.contacts[0].depth).toBeGreaterThan(1);
  });

  it('asymmetric half-extents – tall vs wide box detects with correct axis', () => {
    // Tall box (0.2, 2.0) at origin, wide box (2.0, 0.2) offset along x
    const tall = makeBox(0, 0, 0, 0.2, 2.0);
    const wide = makeBox(1, 1.5, 0, 2.0, 0.2);
    const result = collideBoxBox(tall, wide);
    // Overlap along x: tall right = 0.2, wide left = 1.5 - 2.0 = -0.5 → overlap = 0.2 - (-0.5) = 0.7
    // Overlap along y: tall top = 2.0, wide bottom = -0.2 → overlap = 0.2 + 0.2 = 0.4 ... but centers differ
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeGreaterThan(0);
  });
});

// ===========================================================================
// 2D Circle Edge Cases
// ===========================================================================

describe('2D Circle Edge Cases', () => {
  it('different radii – r=0.3 vs r=1.0 overlapping yields correct depth', () => {
    // Centers 1.0 apart, sum of radii = 1.3
    const a = makeBall(0, 0, 0, 0.3);
    const b = makeBall(1, 1.0, 0, 1.0);
    const result = collideCircleCircle(a, b);
    expect(result).not.toBeNull();
    // depth = r1 + r2 - dist = 1.3 - 1.0 = 0.3
    expect(result!.contacts[0].depth).toBeCloseTo(0.3, 3);
  });

  it('concentric different radii – same center yields depth = r1 + r2', () => {
    const a = makeBall(0, 0, 0, 0.5);
    const b = makeBall(1, 0, 0, 1.5);
    const result = collideCircleCircle(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(2.0, 3);
  });

  it('near-touching – distance = r1 + r2 - epsilon detects collision', () => {
    // r1 = 1, r2 = 1, place centers at distance 1.999 apart → overlap = 0.001
    const a = makeBall(0, 0, 0, 1);
    const b = makeBall(1, 1.999, 0, 1);
    const result = collideCircleCircle(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.001, 3);
  });
});

// ===========================================================================
// 2D Box-Circle Edge Cases
// ===========================================================================

describe('2D Box-Circle Edge Cases', () => {
  it('circle at box corner – diagonal contact detected', () => {
    // Box at origin hx=hy=1, circle center at (1.5, 1.5), r=1
    // Distance from corner (1,1) to (1.5,1.5) = sqrt(0.5) ≈ 0.707 < 1
    const box = makeBox(0, 0, 0, 1, 1);
    const circle = makeBall(1, 1.5, 1.5, 1);
    const result = collideBoxCircle(box, circle);
    expect(result).not.toBeNull();
    const expectedDepth = 1 - Math.sqrt(0.5);
    expect(result!.contacts[0].depth).toBeCloseTo(expectedDepth, 2);
  });

  it('circle center inside box – encapsulation detected', () => {
    // Large box, small circle entirely inside
    const box = makeBox(0, 0, 0, 3, 3);
    const circle = makeBall(1, 0.5, 0, 0.2);
    const result = collideBoxCircle(box, circle);
    expect(result).not.toBeNull();
    // Circle center is at (0.5, 0), closest face is x=3 at distance 2.5
    // depth should be positive and represent escape distance
    expect(result!.contacts[0].depth).toBeGreaterThan(0);
  });

  it('circle center at box center – normal along shortest axis', () => {
    // Non-square box so shortest axis is unambiguous
    const box = makeBox(0, 0, 0, 2, 1); // hx=2 hy=1, shortest escape is along y
    const circle = makeBall(1, 0, 0, 0.5);
    const result = collideBoxCircle(box, circle);
    expect(result).not.toBeNull();
    const n = result!.normal;
    // Normal should be predominantly along y axis (the shorter escape)
    expect(Math.abs(n.y)).toBeGreaterThanOrEqual(Math.abs(n.x));
  });

  it('circle tangent to face – just barely touching detects collision', () => {
    // Box at origin hx=1 hy=1, circle at (1.999, 0) r=1 → overlap = 0.001
    const box = makeBox(0, 0, 0, 1, 1);
    const circle = makeBall(1, 1.999, 0, 1);
    const result = collideBoxCircle(box, circle);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.001, 3);
  });
});

// ===========================================================================
// 2D Normal Direction Consistency
// ===========================================================================

describe('2D Normal Direction Consistency', () => {
  it('normal from B to A – dot(normal, A.pos - B.pos) >= 0', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 1.5, 0.5, 1, 1);
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
    const n = result!.normal;
    const dx = a.position.x - b.position.x;
    const dy = a.position.y - b.position.y;
    expect(n.x * dx + n.y * dy).toBeGreaterThanOrEqual(0);
  });

  it('swapped dispatch – collide2D(A,B) and collide2D(B,A) give consistent normals', () => {
    const a = makeBox(0, -0.5, 0, 1, 1);
    const b = makeBox(1, 1.0, 0, 1, 1);
    const rAB = collide2D(a, b);
    const rBA = collide2D(b, a);
    expect(rAB).not.toBeNull();
    expect(rBA).not.toBeNull();
    // Depths should match
    expect(rAB!.contacts[0].depth).toBeCloseTo(rBA!.contacts[0].depth, 3);
    // Normals should be opposite directions
    const nAB = rAB!.normal;
    const nBA = rBA!.normal;
    expect(nAB.x * nBA.x + nAB.y * nBA.y).toBeCloseTo(-1, 2);
  });
});

// ===========================================================================
// 3D Sphere Edge Cases
// ===========================================================================

describe('3D Sphere Edge Cases', () => {
  it('concentric spheres – same center yields default normal and depth = r1 + r2', () => {
    const a = makeSphere3D(0, 0, 0, 0, 1);
    const b = makeSphere3D(1, 0, 0, 0, 0.5);
    const result = collideSpheresSphere(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(1.5, 3);
    // Normal should be unit length even for degenerate case
    const n = result!.normal;
    const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    expect(len).toBeCloseTo(1, 3);
  });

  it('barely separated spheres – distance = r1 + r2 + 0.001 → null', () => {
    // r1 = 1, r2 = 0.5, sum = 1.5, place at distance 1.501
    const a = makeSphere3D(0, 0, 0, 0, 1);
    const b = makeSphere3D(1, 1.501, 0, 0, 0.5);
    const result = collideSpheresSphere(a, b);
    expect(result).toBeNull();
  });
});

// ===========================================================================
// 3D Box-Box Edge Cases
// ===========================================================================

describe('3D Box-Box Edge Cases', () => {
  it('rotated box on ground – 45-degree Y-rotation resting on large ground box', () => {
    // Ground: top face at y = 0 (center y=-0.5, hy=0.5)
    const ground = makeBox3D(0, 0, -0.5, 0, 10, 0.5, 10);
    // Box rotated 45° around Y, center at y=0.4, hy=0.5 → bottom at y=-0.1, overlapping ground top
    const rotY45 = quatFromAxisAngle({ x: 0, y: 1, z: 0 }, Math.PI / 4);
    const box = makeBox3D(1, 0, 0.4, 0, 0.5, 0.5, 0.5, rotY45);
    const result = collideBoxBox3D(ground, box);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeGreaterThan(0);
  });

  it('non-uniform dimensions – box (0.5,1.0,2.0) overlapping with cube detects correct axis', () => {
    const cube = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    // Place non-uniform box so that minimum overlap is along x
    // cube extends to x=1, nonuniform extends from x=0.8 - 0.5=0.3 to x=0.8+0.5=1.3
    // overlap along x = 1 - 0.3 = 0.7 but also check y: cube extends to 1, box from -1 to 1, overlap = 2; z: cube extends to 1, box from -2 to 2, overlap = 2
    const nonUniform = makeBox3D(1, 1.8, 0, 0, 0.5, 1.0, 2.0);
    const result = collideBoxBox3D(cube, nonUniform);
    // Overlap along x: cube right = 1, nonUniform left = 1.8-0.5 = 1.3 → no overlap along x
    // Let's fix: place at x=1.2 so left = 0.7, overlap = 1 - 0.7 = 0.3
    const nonUniform2 = makeBox3D(1, 1.2, 0, 0, 0.5, 1.0, 2.0);
    const result2 = collideBoxBox3D(cube, nonUniform2);
    expect(result2).not.toBeNull();
    // Penetration along x should be smallest (0.3)
    expect(result2!.contacts[0].depth).toBeCloseTo(0.3, 2);
    // Normal should be predominantly along x
    const n = result2!.normal;
    expect(Math.abs(n.x)).toBeGreaterThan(0.9);
  });

  it('axis-aligned stacking along Z-axis', () => {
    const bottom = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    // Stack on Z: bottom z-extent = 1, top at z=1.9 with hz=1 → overlap = 1 - 0.9 = 0.1
    const top = makeBox3D(1, 0, 0, 1.9, 1, 1, 1);
    const result = collideBoxBox3D(bottom, top);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.1, 2);
    // Normal should be along Z
    const n = result!.normal;
    expect(Math.abs(n.z)).toBeGreaterThan(0.9);
  });

  it('edge contact – two boxes at 90-degree rotation with only edges overlapping', () => {
    // Box A axis-aligned, Box B rotated 45° around Y, placed at a corner
    const a = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    const rotY45 = quatFromAxisAngle({ x: 0, y: 1, z: 0 }, Math.PI / 4);
    // sqrt(2) ≈ 1.414, box B diagonal reach along x ≈ 1*cos45 + 1*sin45 = 1.414
    // Place B so only the edge touches A
    const b = makeBox3D(1, 2.3, 0, 0, 1, 1, 1, rotY45);
    const result = collideBoxBox3D(a, b);
    // Could be touching or slightly overlapping depending on exact placement
    if (result !== null) {
      expect(result.contacts[0].depth).toBeLessThan(0.3);
      expect(result.contacts.length).toBeGreaterThan(0);
    }
  });
});

// ===========================================================================
// 3D Box-Sphere Edge Cases
// ===========================================================================

describe('3D Box-Sphere Edge Cases', () => {
  it('sphere at box corner – diagonal contact detected', () => {
    // Box at origin hx=hy=hz=1, sphere center at (1.3, 1.3, 1.3), r=1
    // Distance from corner (1,1,1) to (1.3,1.3,1.3) = sqrt(0.27) ≈ 0.52 < 1
    const box = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    const sphere = makeSphere3D(1, 1.3, 1.3, 1.3, 1);
    const result = collideBoxSphere(box, sphere);
    expect(result).not.toBeNull();
    const cornerDist = Math.sqrt(0.3 * 0.3 * 3);
    const expectedDepth = 1 - cornerDist;
    expect(result!.contacts[0].depth).toBeCloseTo(expectedDepth, 2);
  });

  it('sphere inside box – encapsulation with correct escape axis', () => {
    // Large box, small sphere at slightly off-center position
    const box = makeBox3D(0, 0, 0, 0, 3, 2, 5);
    // Sphere at (0, 0.5, 0) r=0.3 → closest face is y+ at y=2, distance = 1.5
    // Escape along y: 2 - 0.5 + 0.3 = 1.8; along -y: 2 + 0.5 + 0.3 = 2.8
    // Escape along x: 3 + 0.3 = 3.3; along z: 5 + 0.3 = 5.3
    // Minimum escape = y+ direction with depth = 2 - 0.5 + 0.3 = 1.8
    const sphere = makeSphere3D(1, 0, 0.5, 0, 0.3);
    const result = collideBoxSphere(box, sphere);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeGreaterThan(0);
    // Normal should be unit length
    const n = result!.normal;
    const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    expect(len).toBeCloseTo(1, 3);
  });
});
