/**
 * Spring and motor constraint tests.
 */
import { describe, it, expect } from 'vitest';
import { createSpringConstraintRows, type SpringDef2D } from '../src/constraints/spring.js';
import { createMotorConstraintRows, MotorData2D } from '../src/constraints/motor.js';
import type { Body2D } from '../src/core/rigid-body.js';
import { ColliderShapeType } from '../src/core/rigid-body.js';
import { RigidBodyType, ForceType } from '../src/core/types.js';

function makeBody(index: number, x: number, y: number, angle: number = 0): Body2D {
  return {
    index, type: RigidBodyType.Dynamic,
    position: { x, y }, angle,
    velocity: { x: 0, y: 0 }, angularVelocity: 0,
    mass: 1, invMass: 1, inertia: 0.1, invInertia: 10,
    gravityScale: 1, linearDamping: 0, angularDamping: 0,
    colliderShape: ColliderShapeType.Cuboid,
    halfExtents: { x: 0.5, y: 0.5 }, radius: 0,
    friction: 0.5, restitution: 0.3,
    prevPosition: { x, y }, prevAngle: angle,
    inertialPosition: { x, y }, inertialAngle: angle,
    boundingRadius: 0.707,
  };
}

describe('Spring constraints', () => {
  it('should create a spring constraint row', () => {
    const bodyA = makeBody(0, 0, 0);
    const bodyB = makeBody(1, 3, 0);

    const spring: SpringDef2D = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: 0 },
      localAnchorB: { x: 0, y: 0 },
      restLength: 2,
      stiffness: 1000,
      damping: 10,
    };

    const rows = createSpringConstraintRows(spring, bodyA, bodyB, 100);
    expect(rows.length).toBe(1);
    expect(rows[0].type).toBe(ForceType.Spring);
    // Distance is 3, rest length is 2, so C = 1
    expect(rows[0].c).toBeCloseTo(1);
  });

  it('should have zero constraint error at rest length', () => {
    const bodyA = makeBody(0, 0, 0);
    const bodyB = makeBody(1, 2, 0);

    const spring: SpringDef2D = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: 0 },
      localAnchorB: { x: 0, y: 0 },
      restLength: 2,
      stiffness: 1000,
      damping: 10,
    };

    const rows = createSpringConstraintRows(spring, bodyA, bodyB, 100);
    expect(rows[0].c).toBeCloseTo(0);
  });

  it('should have negative C when compressed', () => {
    const bodyA = makeBody(0, 0, 0);
    const bodyB = makeBody(1, 1, 0);

    const spring: SpringDef2D = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: 0 },
      localAnchorB: { x: 0, y: 0 },
      restLength: 2,
      stiffness: 1000,
      damping: 10,
    };

    const rows = createSpringConstraintRows(spring, bodyA, bodyB, 100);
    expect(rows[0].c).toBeCloseTo(-1); // dist(1) - rest(2) = -1
  });

  it('should cap penalty at stiffness', () => {
    const bodyA = makeBody(0, 0, 0);
    const bodyB = makeBody(1, 3, 0);

    const spring: SpringDef2D = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: 0 },
      localAnchorB: { x: 0, y: 0 },
      restLength: 2,
      stiffness: 50, // Lower than penaltyMin
      damping: 10,
    };

    const rows = createSpringConstraintRows(spring, bodyA, bodyB, 100);
    expect(rows[0].penalty).toBe(50); // Capped at stiffness
  });

  it('should handle rotated anchor points', () => {
    const bodyA = makeBody(0, 0, 0, Math.PI / 2); // 90 degree rotation
    const bodyB = makeBody(1, 3, 0);

    const spring: SpringDef2D = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 1, y: 0 }, // Rotated 90 degrees → (0, 1)
      localAnchorB: { x: 0, y: 0 },
      restLength: 2,
      stiffness: 1000,
      damping: 10,
    };

    const rows = createSpringConstraintRows(spring, bodyA, bodyB, 100);
    expect(rows.length).toBe(1);
    // World anchor A is at (0,0) + rotate(1,0, pi/2) = (0, 1)
    // World anchor B is at (3, 0)
    // Distance = sqrt(9 + 1) ≈ 3.162
    expect(rows[0].c).toBeCloseTo(3.162 - 2, 2);
  });

  it('should return empty for zero-distance bodies', () => {
    const bodyA = makeBody(0, 0, 0);
    const bodyB = makeBody(1, 0, 0); // Same position

    const spring: SpringDef2D = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: 0 },
      localAnchorB: { x: 0, y: 0 },
      restLength: 1,
      stiffness: 1000,
      damping: 10,
    };

    const rows = createSpringConstraintRows(spring, bodyA, bodyB, 100);
    expect(rows.length).toBe(0); // Can't compute direction
  });
});

describe('Motor constraints', () => {
  it('should create a motor constraint row', () => {
    const bodyA = makeBody(0, 0, 0);
    const bodyB = makeBody(1, 0, 0);

    const motor: import('../src/constraints/motor.js').MotorDef2D = {
      bodyA: 0, bodyB: 1,
      targetVelocity: Math.PI,
      maxTorque: 100,
      stiffness: Infinity,
    };

    const rows = createMotorConstraintRows(motor, bodyA, bodyB, 100, 1/60);
    expect(rows.length).toBe(1);
    expect(rows[0].type).toBe(ForceType.Motor);
    // Force bounds should be symmetric and capped at max torque
    expect(rows[0].fmin).toBe(-100);
    expect(rows[0].fmax).toBe(100);
  });

  it('should have angular-only Jacobian', () => {
    const bodyA = makeBody(0, 0, 0);
    const bodyB = makeBody(1, 0, 0);

    const rows = createMotorConstraintRows({
      bodyA: 0, bodyB: 1,
      targetVelocity: 1,
      maxTorque: 50,
      stiffness: Infinity,
    }, bodyA, bodyB, 100, 1/60);

    // Linear components should be zero
    expect(rows[0].jacobianA[0]).toBe(0);
    expect(rows[0].jacobianA[1]).toBe(0);
    expect(rows[0].jacobianA[2]).toBe(1);
    expect(rows[0].jacobianB[2]).toBe(-1);
  });
});

describe('MotorData2D builder', () => {
  it('should create motor data with velocity', () => {
    const md = MotorData2D.velocity(3.14);
    expect(md.targetVelocity).toBeCloseTo(3.14);
    expect(md.maxTorque).toBe(Infinity);
  });

  it('should support chained configuration', () => {
    const md = MotorData2D.velocity(1)
      .setMaxTorque(50)
      .setStiffness(1000);
    expect(md.maxTorque).toBe(50);
    expect(md.stiffness).toBe(1000);
  });
});
