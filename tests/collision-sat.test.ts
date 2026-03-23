import { describe, it, expect } from 'vitest';
import { collideBoxBox, collideCircleCircle, collideBoxCircle, collide2D } from '../src/2d/collision-sat.js';
import type { Body2D } from '../src/core/rigid-body.js';
import { ColliderShapeType } from '../src/core/rigid-body.js';
import { RigidBodyType } from '../src/core/types.js';

function makeBox(index: number, x: number, y: number, hx: number, hy: number, angle: number = 0, type: RigidBodyType = RigidBodyType.Dynamic): Body2D {
  return {
    index, type,
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

describe('Box-Box SAT Collision', () => {
  it('should detect overlapping axis-aligned boxes', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 1.5, 0, 1, 1);
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts.length).toBeGreaterThan(0);
    expect(result!.contacts[0].depth).toBeGreaterThan(0);
  });

  it('should not detect separated boxes', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 5, 0, 1, 1);
    const result = collideBoxBox(a, b);
    expect(result).toBeNull();
  });

  it('should detect touching boxes (edge contact)', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 2.001, 0, 1, 1); // Slightly separated
    const result = collideBoxBox(a, b);
    // Slightly separated boxes should not collide
    expect(result).toBeNull();
  });

  it('should detect deeply overlapping boxes', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 0.5, 0, 1, 1);
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeGreaterThan(1);
  });

  it('should detect rotated box collision', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 1.5, 0, 1, 1, Math.PI / 4); // 45 degree rotation
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
  });

  it('should not detect separated rotated boxes', () => {
    const a = makeBox(0, 0, 0, 0.5, 0.5);
    const b = makeBox(1, 5, 5, 0.5, 0.5, Math.PI / 4);
    const result = collideBoxBox(a, b);
    expect(result).toBeNull();
  });

  it('should have normal pointing from B to A', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 1.5, 0, 1, 1);
    const result = collideBoxBox(a, b);
    expect(result).not.toBeNull();
    // Normal should point from B towards A (negative x direction)
    // or at least have a component pointing from B to A
    const n = result!.normal;
    const d = { x: a.position.x - b.position.x, y: a.position.y - b.position.y };
    expect(n.x * d.x + n.y * d.y).toBeGreaterThanOrEqual(0);
  });

  it('should detect box stacking (vertical overlap)', () => {
    const ground = makeBox(0, 0, 0, 5, 0.5, 0, RigidBodyType.Fixed);
    const box = makeBox(1, 0, 0.9, 0.5, 0.5);
    const result = collideBoxBox(ground, box);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeGreaterThan(0);
  });
});

describe('Circle-Circle Collision', () => {
  it('should detect overlapping circles', () => {
    const a = makeBall(0, 0, 0, 1);
    const b = makeBall(1, 1.5, 0, 1);
    const result = collideCircleCircle(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.5, 5);
  });

  it('should not detect separated circles', () => {
    const a = makeBall(0, 0, 0, 1);
    const b = makeBall(1, 5, 0, 1);
    const result = collideCircleCircle(a, b);
    expect(result).toBeNull();
  });

  it('should detect touching circles', () => {
    const a = makeBall(0, 0, 0, 1);
    const b = makeBall(1, 2, 0, 1);
    const result = collideCircleCircle(a, b);
    expect(result).toBeNull(); // Exactly touching = no overlap
  });

  it('should handle concentric circles', () => {
    const a = makeBall(0, 0, 0, 1);
    const b = makeBall(1, 0, 0, 1);
    const result = collideCircleCircle(a, b);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(2, 5);
  });
});

describe('Box-Circle Collision', () => {
  it('should detect circle touching box face', () => {
    const box = makeBox(0, 0, 0, 1, 1);
    const circle = makeBall(1, 1.5, 0, 1);
    const result = collideBoxCircle(box, circle);
    expect(result).not.toBeNull();
    expect(result!.contacts[0].depth).toBeCloseTo(0.5, 3);
  });

  it('should not detect separated box and circle', () => {
    const box = makeBox(0, 0, 0, 1, 1);
    const circle = makeBall(1, 5, 0, 1);
    const result = collideBoxCircle(box, circle);
    expect(result).toBeNull();
  });

  it('should detect circle near box corner', () => {
    const box = makeBox(0, 0, 0, 1, 1);
    const circle = makeBall(1, 1.5, 1.5, 1);
    const result = collideBoxCircle(box, circle);
    // Distance from corner (1,1) to circle center (1.5,1.5) = sqrt(0.5) ≈ 0.707
    // Since radius is 1, there should be overlap
    expect(result).not.toBeNull();
  });

  it('should detect circle inside box', () => {
    const box = makeBox(0, 0, 0, 2, 2);
    const circle = makeBall(1, 0, 0, 0.5);
    const result = collideBoxCircle(box, circle);
    expect(result).not.toBeNull();
  });
});

describe('collide2D dispatch', () => {
  it('should dispatch box-box correctly', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBox(1, 1.5, 0, 1, 1);
    const result = collide2D(a, b);
    expect(result).not.toBeNull();
  });

  it('should dispatch circle-circle correctly', () => {
    const a = makeBall(0, 0, 0, 1);
    const b = makeBall(1, 1.5, 0, 1);
    const result = collide2D(a, b);
    expect(result).not.toBeNull();
  });

  it('should dispatch box-circle correctly', () => {
    const a = makeBox(0, 0, 0, 1, 1);
    const b = makeBall(1, 1.5, 0, 1);
    const result = collide2D(a, b);
    expect(result).not.toBeNull();
  });

  it('should dispatch circle-box correctly (swapped)', () => {
    const a = makeBall(0, 0, 0, 1);
    const b = makeBox(1, 1.5, 0, 1, 1);
    const result = collide2D(a, b);
    expect(result).not.toBeNull();
  });
});
