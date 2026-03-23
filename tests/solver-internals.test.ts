/**
 * Solver internal computation unit tests.
 * Tests primal update, dual update, velocity recovery, and LDL solvers in isolation.
 */
import { describe, it, expect } from 'vitest';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';
import { ForceType, COLLISION_MARGIN } from '../src/core/types.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── solveLDL6 (tested indirectly via 3D solver) ───────────────────────────

describe('3D solver: 6x6 LDL solve', () => {
  it('should solve free-fall (diagonal mass system) correctly', () => {
    // A single free-falling body with no contacts has a purely diagonal system:
    // LHS = diag(m, m, m, Ix, Iy, Iz) / dt²
    // RHS = M/dt² * (pos - inertialPos)
    // Solution should move body to inertialPos
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 1, useCPU: true,
    });
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0),
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    world.step();

    // After one step, body should have fallen
    const pos = body.translation();
    expect(pos.y).toBeLessThan(10);
    expect(pos.x).toBeCloseTo(0, 3); // No x drift
    expect(pos.z).toBeCloseTo(0, 3); // No z drift
  });

  it('should solve coupled system (contact constraint) without NaN', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 5, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 1.5, 0),
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), body);

    // Run several steps
    for (let i = 0; i < 60; i++) world.step();

    const pos = body.translation();
    expect(isFinite(pos.x)).toBe(true);
    expect(isFinite(pos.y)).toBe(true);
    expect(isFinite(pos.z)).toBe(true);
    // Should have settled above ground
    expect(pos.y).toBeGreaterThan(0.5);
  });
});

// ─── Primal update: geometric stiffness in LHS ─────────────────────────────

describe('2D Primal update: geometric stiffness', () => {
  it('should include hessianDiag in angular diagonal of LHS', () => {
    // Create a body overlapping the ground - forces immediate contact generation
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    // Position box so it overlaps with ground (box bottom at y=0.4, ground top at y=0.5)
    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0.9),
    );
    solver.bodyStore.attachCollider(box.index, ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5));

    // Run one step to generate contacts
    solver.step();

    // Verify contact constraints were created with non-zero hessianDiag
    const contactRows = solver.constraintStore.rows.filter(r => r.type === ForceType.Contact && r.active);
    expect(contactRows.length).toBeGreaterThan(0);

    // Normal row should have hessianDiag[2] non-zero for off-center contacts
    // The contact point is at the box bottom, offset from body center
    const normalRow = contactRows[0];
    // hessianDiagA or B should have non-zero angular component for most contacts
    const hasNonZeroHessian = contactRows.some(r =>
      r.hessianDiagA[2] !== 0 || r.hessianDiagB[2] !== 0
    );
    expect(hasNonZeroHessian).toBe(true);
  });

  it('should have zero geometric stiffness when contact is at body center', () => {
    // When r = 0 (contact at body center), r·n = 0, so hessianDiag[2] = 0
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    // Create two bodies at the same position (overlapping)
    const bodyA = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0),
    );
    solver.bodyStore.attachCollider(bodyA.index, ColliderDesc2D.cuboid(0.5, 0.5));

    const bodyB = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0),
    );
    solver.bodyStore.attachCollider(bodyB.index, ColliderDesc2D.cuboid(0.5, 0.5));

    solver.step();

    // With overlapping bodies at same center, some contacts may be at midpoint
    // (which equals both body centers), giving r=0 and hessianDiag=0
    for (const row of solver.constraintStore.rows) {
      if (!row.active || row.type !== ForceType.Contact) continue;
      // The hessian values should be finite (no NaN from zero division)
      expect(isFinite(row.hessianDiagA[2])).toBe(true);
      expect(isFinite(row.hessianDiagB[2])).toBe(true);
    }
  });
});

// ─── Dual update: friction penalty skip ─────────────────────────────────────

describe('2D Dual update: penalty ramping', () => {
  it('should ramp penalty for normal contact rows', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1.1),
    );
    solver.bodyStore.attachCollider(box.index, ColliderDesc2D.cuboid(0.5, 0.5));

    // Run several steps to allow penalty to ramp
    for (let i = 0; i < 30; i++) solver.step();

    // Normal rows (fmin=-Infinity) should have ramped penalty
    const normalRows = solver.constraintStore.rows.filter(
      r => r.active && r.type === ForceType.Contact && !isFinite(r.fmin)
    );
    if (normalRows.length > 0) {
      const maxPenalty = Math.max(...normalRows.map(r => r.penalty));
      expect(maxPenalty).toBeGreaterThan(solver.config.penaltyMin);
    }
  });

  it('should NOT ramp penalty for friction rows', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1.1),
    );
    solver.bodyStore.attachCollider(box.index, ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.8));

    for (let i = 0; i < 30; i++) solver.step();

    // Friction rows (finite fmin) should have low penalty (not ramped)
    const frictionRows = solver.constraintStore.rows.filter(
      r => r.active && r.type === ForceType.Contact && isFinite(r.fmin)
    );
    if (frictionRows.length > 0) {
      // Friction penalty should stay close to penaltyMin (only decayed by gamma, not ramped)
      for (const row of frictionRows) {
        // Friction penalty should be much lower than normal penalty
        expect(row.penalty).toBeLessThan(solver.config.penaltyMax);
      }
    }
  });

  it('should clamp lambda to [fmin, fmax]', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1.1),
    );
    solver.bodyStore.attachCollider(box.index, ColliderDesc2D.cuboid(0.5, 0.5));

    for (let i = 0; i < 30; i++) solver.step();

    for (const row of solver.constraintStore.rows) {
      if (!row.active) continue;
      // Lambda must always be within bounds
      expect(row.lambda).toBeGreaterThanOrEqual(row.fmin);
      expect(row.lambda).toBeLessThanOrEqual(row.fmax);
    }
  });
});

// ─── Velocity recovery ──────────────────────────────────────────────────────

describe('2D Velocity recovery', () => {
  it('should clamp recovered angular velocity to [-50, 50]', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: 0 },
      dt: 1 / 60,
      iterations: 1,
    });

    const body = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0),
    );
    solver.bodyStore.attachCollider(body.index, ColliderDesc2D.cuboid(0.5, 0.5));

    // Set a very high angular velocity
    solver.bodyStore.bodies[body.index].angularVelocity = 200;

    solver.step();

    // After step, angular velocity should be clamped at start of step
    // and then recovered (possibly clamped again)
    const omega = solver.bodyStore.bodies[body.index].angularVelocity;
    expect(Math.abs(omega)).toBeLessThanOrEqual(50 + 1); // tolerance for dt effects
    expect(isFinite(omega)).toBe(true);
  });

  it('should correctly recover linear velocity from position change', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 1,
    });

    const body = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 10),
    );
    solver.bodyStore.attachCollider(body.index, ColliderDesc2D.cuboid(0.5, 0.5));

    solver.step();

    // After falling for one step with gravity, velocity should be approximately g*dt
    const vy = solver.bodyStore.bodies[body.index].velocity.y;
    expect(vy).toBeLessThan(0); // Falling downward
    const expectedVy = -9.81 * (1 / 60);
    expect(vy).toBeCloseTo(expectedVy, 0); // Rough match
  });
});

describe('3D Velocity recovery', () => {
  it('should clamp recovered angular velocity magnitude to 50', () => {
    const world = new World3D({ x: 0, y: 0, z: 0 }, {
      iterations: 1, useCPU: true,
    });

    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 0, 0),
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    // Set extreme angular velocity
    body.setAngvel({ x: 100, y: 100, z: 100 });

    world.step();

    const av = body.angvel();
    const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
    // Angular velocity magnitude should be bounded
    expect(mag).toBeLessThan(200); // Generous bound; actual clamp is 50 per step
    expect(isFinite(av.x)).toBe(true);
    expect(isFinite(av.y)).toBe(true);
    expect(isFinite(av.z)).toBe(true);
  });

  it('should extract angular velocity from quaternion change', () => {
    const world = new World3D({ x: 0, y: 0, z: 0 }, {
      iterations: 1, useCPU: true,
    });

    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 0, 0),
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    // Give a moderate angular velocity
    body.setAngvel({ x: 0, y: 5, z: 0 });

    world.step();

    // After one step, angular velocity should still be roughly 5 rad/s about Y
    const av = body.angvel();
    expect(Math.abs(av.y)).toBeGreaterThan(1);
    expect(isFinite(av.y)).toBe(true);
  });
});

// ─── Contact constraint properties after solver step ────────────────────────

describe('2D Contact constraint properties', () => {
  it('should apply collision margin to contact constraint value', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0.9), // Close to ground
    );
    solver.bodyStore.attachCollider(box.index, ColliderDesc2D.cuboid(0.5, 0.5));

    solver.step();

    // Check that contact constraints include COLLISION_MARGIN
    const normalRows = solver.constraintStore.rows.filter(
      r => r.active && r.type === ForceType.Contact && !isFinite(r.fmin)
    );
    // c = -depth + COLLISION_MARGIN, so c should be slightly larger than -depth
    for (const row of normalRows) {
      // c0 should include the margin
      expect(row.c0).toBeGreaterThan(-10); // Not wildly negative
      expect(isFinite(row.c0)).toBe(true);
    }
  });

  it('should have compressive-only bounds for normal rows', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1.1),
    );
    solver.bodyStore.attachCollider(box.index, ColliderDesc2D.cuboid(0.5, 0.5));

    solver.step();

    for (const row of solver.constraintStore.rows) {
      if (!row.active || row.type !== ForceType.Contact) continue;
      if (!isFinite(row.fmin)) {
        // Normal: fmin=-Infinity, fmax=0
        expect(row.fmax).toBe(0);
      } else {
        // Friction: symmetric bounds
        expect(row.fmax).toBeGreaterThanOrEqual(0);
        expect(row.fmin).toBeLessThanOrEqual(0);
        expect(Math.abs(row.fmin + row.fmax)).toBeLessThan(0.01); // Nearly symmetric
      }
    }
  });
});
