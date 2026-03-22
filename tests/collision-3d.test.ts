import { describe, it, expect } from 'vitest';
import { collideBoxBox3D, collideSpheresSphere, collideBoxSphere, collide3D, getAABB3D, aabb3DOverlap } from '../src/3d/collision-gjk.js';
import type { Body3D } from '../src/core/rigid-body-3d.js';
import { ColliderShapeType3D } from '../src/core/rigid-body-3d.js';
import { RigidBodyType } from '../src/core/types.js';
import { quatIdentity, quatFromAxisAngle, vec3 } from '../src/core/math.js';

function makeBox3D(index: number, x: number, y: number, z: number, hx: number, hy: number, hz: number): Body3D {
  return {
    index, type: RigidBodyType.Dynamic,
    position: { x, y, z }, rotation: quatIdentity(),
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

describe('3D AABB', () => {
  it('should compute AABB for axis-aligned box', () => {
    const b = makeBox3D(0, 1, 2, 3, 0.5, 0.5, 0.5);
    const aabb = getAABB3D(b);
    expect(aabb.min.x).toBeCloseTo(0.5);
    expect(aabb.max.x).toBeCloseTo(1.5);
    expect(aabb.min.y).toBeCloseTo(1.5);
    expect(aabb.max.y).toBeCloseTo(2.5);
  });

  it('should compute AABB for sphere', () => {
    const b = makeSphere3D(0, 1, 2, 3, 0.5);
    const aabb = getAABB3D(b);
    expect(aabb.min.x).toBeCloseTo(0.5);
    expect(aabb.max.x).toBeCloseTo(1.5);
  });

  it('should detect overlapping AABBs', () => {
    const a = getAABB3D(makeBox3D(0, 0, 0, 0, 1, 1, 1));
    const b = getAABB3D(makeBox3D(1, 1.5, 0, 0, 1, 1, 1));
    expect(aabb3DOverlap(a, b)).toBe(true);
  });

  it('should detect separated AABBs', () => {
    const a = getAABB3D(makeBox3D(0, 0, 0, 0, 1, 1, 1));
    const b = getAABB3D(makeBox3D(1, 5, 0, 0, 1, 1, 1));
    expect(aabb3DOverlap(a, b)).toBe(false);
  });
});

describe('Sphere-Sphere 3D', () => {
  it('should detect overlapping spheres', () => {
    const a = makeSphere3D(0, 0, 0, 0, 1);
    const b = makeSphere3D(1, 1.5, 0, 0, 1);
    const result = collideSpheresSphere(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.5);
  });

  it('should not detect separated spheres', () => {
    const a = makeSphere3D(0, 0, 0, 0, 1);
    const b = makeSphere3D(1, 5, 0, 0, 1);
    expect(collideSpheresSphere(a, b)).toBeNull();
  });
});

describe('Box-Box 3D SAT', () => {
  it('should detect overlapping boxes', () => {
    const a = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    const b = makeBox3D(1, 1.5, 0, 0, 1, 1, 1);
    const result = collideBoxBox3D(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.5);
  });

  it('should not detect separated boxes', () => {
    const a = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    const b = makeBox3D(1, 5, 0, 0, 1, 1, 1);
    expect(collideBoxBox3D(a, b)).toBeNull();
  });

  it('should detect box stacking', () => {
    const ground = makeBox3D(0, 0, 0, 0, 5, 0.5, 5);
    const box = makeBox3D(1, 0, 0.9, 0, 0.5, 0.5, 0.5);
    const result = collideBoxBox3D(ground, box);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.1);
  });
});

describe('Box-Sphere 3D', () => {
  it('should detect sphere touching box face', () => {
    const box = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    const sphere = makeSphere3D(1, 1.5, 0, 0, 1);
    const result = collideBoxSphere(box, sphere);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.5);
  });

  it('should not detect separated box and sphere', () => {
    const box = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    const sphere = makeSphere3D(1, 5, 0, 0, 1);
    expect(collideBoxSphere(box, sphere)).toBeNull();
  });
});

describe('collide3D dispatch', () => {
  it('should dispatch all shape combinations', () => {
    const box1 = makeBox3D(0, 0, 0, 0, 1, 1, 1);
    const box2 = makeBox3D(1, 1.5, 0, 0, 1, 1, 1);
    expect(collide3D(box1, box2)).not.toBeNull();

    const s1 = makeSphere3D(0, 0, 0, 0, 1);
    const s2 = makeSphere3D(1, 1.5, 0, 0, 1);
    expect(collide3D(s1, s2)).not.toBeNull();

    expect(collide3D(box1, s2)).not.toBeNull();
    expect(collide3D(s1, box2)).not.toBeNull();
  });
});
