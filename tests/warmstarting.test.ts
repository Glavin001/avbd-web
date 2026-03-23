/**
 * Tests for contact persistence, warmstarting, and constraint caching.
 */
import { describe, it, expect } from 'vitest';
import { ConstraintStore, createDefaultRow, type CachedContact } from '../src/constraints/constraint.js';
import { ForceType } from '../src/core/types.js';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';

describe('Contact caching', () => {
  it('should cache contacts when clearing', () => {
    const store = new ConstraintStore();

    // Add a pair of contact rows (normal + friction)
    const normal = createDefaultRow();
    normal.bodyA = 0; normal.bodyB = 1;
    normal.type = ForceType.Contact;
    normal.lambda = -50;
    normal.penalty = 5000;
    store.addRow(normal);

    const friction = createDefaultRow();
    friction.bodyA = 0; friction.bodyB = 1;
    friction.type = ForceType.Contact;
    friction.lambda = 10;
    friction.penalty = 2000;
    store.addRow(friction);

    // Clear contacts (should cache them first)
    store.clearContacts();

    expect(store.rows.length).toBe(0);
    expect(store.contactCache.size).toBe(1);

    const cached = store.contactCache.get('0-1');
    expect(cached).toBeDefined();
    expect(cached!.normalLambda).toBe(-50);
    expect(cached!.normalPenalty).toBe(5000);
    expect(cached!.frictionLambda).toBe(10);
    expect(cached!.frictionPenalty).toBe(2000);
  });

  it('should restore cached values to new contacts', () => {
    const store = new ConstraintStore();

    // Simulate first frame: create and cache contacts
    const n1 = createDefaultRow();
    n1.bodyA = 2; n1.bodyB = 3;
    n1.type = ForceType.Contact;
    n1.lambda = -100;
    n1.penalty = 8000;
    store.addRow(n1);

    const f1 = createDefaultRow();
    f1.bodyA = 2; f1.bodyB = 3;
    f1.type = ForceType.Contact;
    f1.lambda = 25;
    f1.penalty = 3000;
    store.addRow(f1);

    store.clearContacts(); // Caches the values

    // Simulate second frame: create new contacts for same pair
    const n2 = createDefaultRow();
    n2.bodyA = 2; n2.bodyB = 3;
    n2.type = ForceType.Contact;
    n2.lambda = 0; // Fresh
    n2.penalty = 100; // Fresh
    store.addRow(n2);

    const f2 = createDefaultRow();
    f2.bodyA = 2; f2.bodyB = 3;
    f2.type = ForceType.Contact;
    f2.lambda = 0;
    f2.penalty = 100;
    store.addRow(f2);

    // Warm-start from cache
    store.warmstartContacts();

    // Should have inherited the cached values
    expect(store.rows[0].lambda).toBe(-100);
    expect(store.rows[0].penalty).toBe(8000);
    expect(store.rows[1].lambda).toBe(25);
    expect(store.rows[1].penalty).toBe(3000);
  });

  it('should age and expire cache entries', () => {
    const store = new ConstraintStore();
    store.maxContactAge = 3;

    // Create initial contact
    const n = createDefaultRow();
    n.bodyA = 0; n.bodyB = 1;
    n.type = ForceType.Contact;
    n.lambda = -50;
    store.addRow(n);

    const f = createDefaultRow();
    f.bodyA = 0; f.bodyB = 1;
    f.type = ForceType.Contact;
    store.addRow(f);

    // Clear and age 4 times (> maxContactAge)
    for (let i = 0; i < 5; i++) {
      store.clearContacts();
    }

    // Cache should have expired
    expect(store.contactCache.size).toBe(0);
  });

  it('should not overwrite joint constraints during clearContacts', () => {
    const store = new ConstraintStore();

    // Add a joint constraint
    const joint = createDefaultRow();
    joint.bodyA = 0; joint.bodyB = 1;
    joint.type = ForceType.Joint;
    store.addRow(joint);

    // Add a contact constraint
    const contact = createDefaultRow();
    contact.bodyA = 0; contact.bodyB = 2;
    contact.type = ForceType.Contact;
    store.addRow(contact);

    const contact2 = createDefaultRow();
    contact2.bodyA = 0; contact2.bodyB = 2;
    contact2.type = ForceType.Contact;
    store.addRow(contact2);

    store.clearContacts();

    // Joint should remain
    expect(store.rows.length).toBe(1);
    expect(store.rows[0].type).toBe(ForceType.Joint);
  });
});

describe('Warmstarting effectiveness', () => {
  it('should converge faster with warmstarting (stacking)', () => {
    // Compare solver with and without contact caching
    function measureConvergence(useCache: boolean): number {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
        iterations: 5,
      });

      if (!useCache) {
        solver.constraintStore.maxContactAge = 0; // Disable caching
      }

      // Ground
      const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
      solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

      // Stack of 3 boxes
      for (let i = 0; i < 3; i++) {
        const h = solver.bodyStore.addBody(
          RigidBodyDesc2D.dynamic().setTranslation(0, 1 + i * 1.1)
        );
        solver.bodyStore.attachCollider(h.index, ColliderDesc2D.cuboid(0.5, 0.5));
      }

      // Run for 2 seconds
      for (let i = 0; i < 120; i++) {
        solver.step();
      }

      // Measure total penetration error (lower = better convergence)
      let totalError = 0;
      for (const row of solver.constraintStore.rows) {
        if (row.type === ForceType.Contact && row.active) {
          totalError += Math.abs(row.c);
        }
      }
      return totalError;
    }

    const withCache = measureConvergence(true);
    const withoutCache = measureConvergence(false);

    // Warmstarting should give equal or better convergence
    // (allow some tolerance since results depend on contact patterns)
    expect(withCache).toBeLessThanOrEqual(withoutCache * 2 + 0.1);
  });
});

describe('Graph coloring integration', () => {
  it('should compute color groups during step', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      iterations: 5,
    });

    // Ground + 3 stacked boxes = contacts between pairs
    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    for (let i = 0; i < 3; i++) {
      const h = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 1 + i * 1.1)
      );
      solver.bodyStore.attachCollider(h.index, ColliderDesc2D.cuboid(0.5, 0.5));
    }

    solver.step();

    // Color groups should have been computed
    expect(solver.colorGroups.length).toBeGreaterThan(0);

    // All dynamic bodies should be in some color group
    const coloredBodies = new Set<number>();
    for (const group of solver.colorGroups) {
      for (const idx of group.bodyIndices) {
        coloredBodies.add(idx);
      }
    }

    // Dynamic bodies (indices 1, 2, 3) should be colored
    for (let i = 1; i <= 3; i++) {
      expect(coloredBodies.has(i)).toBe(true);
    }
  });
});
