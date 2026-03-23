import { describe, it, expect } from 'vitest';
import { ConstraintStore, createDefaultRow } from '../src/constraints/constraint.js';
import { createContactConstraintRows } from '../src/constraints/contact.js';
import { createJointConstraintRows, JointData2D } from '../src/constraints/joint.js';
import { ForceType, RigidBodyType } from '../src/core/types.js';
import type { Body2D } from '../src/core/rigid-body.js';
import { ColliderShapeType } from '../src/core/rigid-body.js';

function makeBody(index: number, x: number, y: number, angle: number = 0, type: RigidBodyType = RigidBodyType.Dynamic): Body2D {
  return {
    index, type,
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

describe('ConstraintStore', () => {
  it('should add and retrieve constraint rows', () => {
    const store = new ConstraintStore();
    const row = createDefaultRow();
    row.bodyA = 0;
    row.bodyB = 1;
    const idx = store.addRow(row);
    expect(idx).toBe(0);
    expect(store.count).toBe(1);
    expect(store.activeCount).toBe(1);
  });

  it('should get rows for a specific body', () => {
    const store = new ConstraintStore();

    const r1 = createDefaultRow();
    r1.bodyA = 0; r1.bodyB = 1;
    store.addRow(r1);

    const r2 = createDefaultRow();
    r2.bodyA = 1; r2.bodyB = 2;
    store.addRow(r2);

    const r3 = createDefaultRow();
    r3.bodyA = 0; r3.bodyB = 3;
    store.addRow(r3);

    const rowsFor0 = store.getRowsForBody(0);
    expect(rowsFor0.length).toBe(2);

    const rowsFor1 = store.getRowsForBody(1);
    expect(rowsFor1.length).toBe(2);

    const rowsFor2 = store.getRowsForBody(2);
    expect(rowsFor2.length).toBe(1);
  });

  it('should get constraint pairs for graph coloring', () => {
    const store = new ConstraintStore();

    const r1 = createDefaultRow();
    r1.bodyA = 0; r1.bodyB = 1;
    store.addRow(r1);

    const r2 = createDefaultRow();
    r2.bodyA = 0; r2.bodyB = 1; // Duplicate pair
    store.addRow(r2);

    const r3 = createDefaultRow();
    r3.bodyA = 1; r3.bodyB = 2;
    store.addRow(r3);

    const pairs = store.getConstraintPairs();
    expect(pairs.length).toBe(2); // Deduplicates
  });

  it('should clear contact constraints but keep joints', () => {
    const store = new ConstraintStore();

    const contact = createDefaultRow();
    contact.bodyA = 0; contact.bodyB = 1;
    contact.type = ForceType.Contact;
    store.addRow(contact);

    const joint = createDefaultRow();
    joint.bodyA = 0; joint.bodyB = 2;
    joint.type = ForceType.Joint;
    store.addRow(joint);

    expect(store.count).toBe(2);
    store.clearContacts();
    expect(store.count).toBe(1);
    expect(store.rows[0].type).toBe(ForceType.Joint);
  });
});

describe('Contact constraint creation', () => {
  it('should create normal + friction rows per contact', () => {
    const bodyA = makeBody(0, 0, 0, 0, RigidBodyType.Fixed);
    const bodyB = makeBody(1, 0, 1);

    const manifold = {
      bodyA: 0,
      bodyB: 1,
      normal: { x: 0, y: -1 },
      contacts: [{ position: { x: 0, y: 0.5 }, normal: { x: 0, y: -1 }, depth: 0.1 }],
    };

    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows.length).toBe(2); // 1 normal + 1 friction

    // Normal row
    expect(rows[0].fmin).toBe(-Infinity);
    expect(rows[0].fmax).toBe(0);
    expect(rows[0].c).toBeCloseTo(-0.1);

    // Friction row
    expect(rows[1].fmin).toBeLessThan(0);
    expect(rows[1].fmax).toBeGreaterThan(0);
  });

  it('should create 4 rows for 2 contact points', () => {
    const bodyA = makeBody(0, 0, 0, 0, RigidBodyType.Fixed);
    const bodyB = makeBody(1, 0, 1);

    const manifold = {
      bodyA: 0, bodyB: 1,
      normal: { x: 0, y: -1 },
      contacts: [
        { position: { x: -0.5, y: 0.5 }, normal: { x: 0, y: -1 }, depth: 0.1 },
        { position: { x: 0.5, y: 0.5 }, normal: { x: 0, y: -1 }, depth: 0.1 },
      ],
    };

    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows.length).toBe(4); // 2 normal + 2 friction
  });
});

describe('Joint constraint creation', () => {
  it('should create revolute joint rows (2 positional)', () => {
    const bodyA = makeBody(0, 0, 2);
    const bodyB = makeBody(1, 0, 0);

    const joint = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: -1 },
      localAnchorB: { x: 0, y: 1 },
      angleConstraint: false,
      targetAngle: 0,
      stiffness: Infinity,
      fractureThreshold: 0,
    };

    const rows = createJointConstraintRows(joint, bodyA, bodyB, 100);
    expect(rows.length).toBe(2); // X and Y constraints
    expect(rows[0].type).toBe(ForceType.Joint);
    expect(rows[1].type).toBe(ForceType.Joint);
  });

  it('should create fixed joint rows (3 = 2 positional + 1 angle)', () => {
    const bodyA = makeBody(0, 0, 2);
    const bodyB = makeBody(1, 0, 0);

    const joint = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: -1 },
      localAnchorB: { x: 0, y: 1 },
      angleConstraint: true,
      targetAngle: 0,
      stiffness: Infinity,
      fractureThreshold: 0,
    };

    const rows = createJointConstraintRows(joint, bodyA, bodyB, 100);
    expect(rows.length).toBe(3);
  });

  it('should have zero constraint error when bodies are aligned', () => {
    const bodyA = makeBody(0, 0, 1);
    const bodyB = makeBody(1, 0, 0);

    const joint = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: -0.5 },
      localAnchorB: { x: 0, y: 0.5 },
      angleConstraint: false,
      targetAngle: 0,
      stiffness: Infinity,
      fractureThreshold: 0,
    };

    const rows = createJointConstraintRows(joint, bodyA, bodyB, 100);
    // Anchor A world = (0, 1) + rotate(0, -0.5, 0) = (0, 0.5)
    // Anchor B world = (0, 0) + rotate(0, 0.5, 0) = (0, 0.5)
    // Error should be zero
    expect(rows[0].c).toBeCloseTo(0);
    expect(rows[1].c).toBeCloseTo(0);
  });

  it('should have non-zero constraint error when misaligned', () => {
    const bodyA = makeBody(0, 0, 2);
    const bodyB = makeBody(1, 1, 0); // offset in x

    const joint = {
      bodyA: 0, bodyB: 1,
      localAnchorA: { x: 0, y: -1 },
      localAnchorB: { x: 0, y: 1 },
      angleConstraint: false,
      targetAngle: 0,
      stiffness: Infinity,
      fractureThreshold: 0,
    };

    const rows = createJointConstraintRows(joint, bodyA, bodyB, 100);
    // Anchor A world = (0, 2) + (0, -1) = (0, 1)
    // Anchor B world = (1, 0) + (0, 1) = (1, 1)
    // Error: (0-1, 1-1) = (-1, 0)
    expect(rows[0].c).toBeCloseTo(-1); // X error
    expect(rows[1].c).toBeCloseTo(0);  // Y error
  });
});

describe('JointData2D builder', () => {
  it('should create revolute joint data', () => {
    const jd = JointData2D.revolute({ x: 0, y: -1 }, { x: 0, y: 1 });
    expect(jd.localAnchorA).toEqual({ x: 0, y: -1 });
    expect(jd.localAnchorB).toEqual({ x: 0, y: 1 });
    expect(jd.angleConstraint).toBe(false);
    expect(jd.stiffness).toBe(Infinity);
  });

  it('should create fixed joint data', () => {
    const jd = JointData2D.fixed({ x: 0, y: 0 }, { x: 0, y: 0 });
    expect(jd.angleConstraint).toBe(true);
  });

  it('should support chained configuration', () => {
    const jd = JointData2D.revolute({ x: 0, y: 0 }, { x: 0, y: 0 })
      .setStiffness(1000)
      .setFractureThreshold(500);
    expect(jd.stiffness).toBe(1000);
    expect(jd.fractureThreshold).toBe(500);
  });
});
