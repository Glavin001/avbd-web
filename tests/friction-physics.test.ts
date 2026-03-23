/**
 * Friction behavior tests.
 * Validates that friction correctly affects sliding, stopping, and deceleration.
 */
import { describe, it, expect } from 'vitest';
import { World, RigidBodyDesc2D, ColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── 2D Friction ────────────────────────────────────────────────────────────

describe('2D Friction correctness', () => {
  it('low friction slides farther than high friction', () => {
    function slideDistance(friction: number): number {
      const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
      world.createCollider(ColliderDesc2D.cuboid(20, 0.5).setFriction(friction));
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 1.1).setLinvel(5, 0)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3).setFriction(friction), body);
      for (let i = 0; i < 120; i++) world.stepCPU();
      return body.translation().x;
    }

    const lowFric = slideDistance(0.1);
    const highFric = slideDistance(1.0);

    expect(lowFric).toBeGreaterThan(0);
    expect(highFric).toBeGreaterThan(-1); // May have barely moved or stopped
    expect(isFinite(lowFric)).toBe(true);
    expect(isFinite(highFric)).toBe(true);
    expect(lowFric).toBeGreaterThan(highFric);
  });

  it('zero friction preserves horizontal velocity', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5).setFriction(0));
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1.1).setLinvel(5, 0)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3).setFriction(0).setRestitution(0), body);

    for (let i = 0; i < 120; i++) world.stepCPU();

    // With zero friction, horizontal velocity should be mostly preserved
    expect(body.linvel().x).toBeGreaterThan(2); // Allow some numerical damping
  });

  it('high friction stops sliding box', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5).setFriction(2.0));
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1.1).setLinvel(5, 0)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3).setFriction(2.0).setRestitution(0), body);

    for (let i = 0; i < 180; i++) world.stepCPU();

    // High friction should significantly slow the body
    expect(Math.abs(body.linvel().x)).toBeLessThan(5);
  });
});

// ─── 3D Friction ────────────────────────────────────────────────────────────

describe('3D Friction correctness', () => {
  it('friction decelerates sliding box', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.8));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 1.5, 0).setLinvel(5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.8), body);

    for (let i = 0; i < 120; i++) world.step();

    expect(Math.abs(body.linvel().x)).toBeLessThan(5);
  });

  it('zero friction allows persistent sliding', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
    world.createCollider(ColliderDesc3D.cuboid(20, 0.5, 20).setFriction(0));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 1.5, 0).setLinvel(5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0), body);

    for (let i = 0; i < 120; i++) world.step();

    // With zero friction, some velocity should persist
    expect(body.linvel().x).toBeGreaterThan(1);
  });

  it('friction works in Z direction same as X', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true, iterations: 10 });
    world.createCollider(ColliderDesc3D.cuboid(20, 0.5, 20).setFriction(0.5));

    const bodyX = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(-5, 1.5, 0).setLinvel(5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), bodyX);

    const bodyZ = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(5, 1.5, 0).setLinvel(0, 0, 5)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), bodyZ);

    for (let i = 0; i < 120; i++) world.step();

    const speedX = Math.abs(bodyX.linvel().x);
    const speedZ = Math.abs(bodyZ.linvel().z);
    // Both should decelerate similarly
    expect(Math.abs(speedX - speedZ)).toBeLessThan(2.0);
  });

  it('low friction slides farther than high friction (3D)', () => {
    function slideDistance3D(friction: number): number {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
      world.createCollider(ColliderDesc3D.cuboid(20, 0.5, 20).setFriction(friction));
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 1.5, 0).setLinvel(5, 0, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(friction), body);
      for (let i = 0; i < 120; i++) world.step();
      return body.translation().x;
    }

    const lowFric = slideDistance3D(0.1);
    const highFric = slideDistance3D(1.0);

    expect(isFinite(lowFric)).toBe(true);
    expect(isFinite(highFric)).toBe(true);
    expect(lowFric).toBeGreaterThan(highFric);
  });
});

// ─── Coulomb Cone Bounds ────────────────────────────────────────────────────

describe('Friction: Coulomb cone bounds', () => {
  it('friction lambda should be bounded by mu * normal lambda (2D)', async () => {
    // Use the raw solver to inspect constraint lambdas directly
    const { AVBDSolver2D: Solver2D } = await import('../src/core/solver.js');
    const { RigidBodyDesc2D: Desc2D, ColliderDesc2D: Coll2D } = await import('../src/core/rigid-body.js');
    const { ForceType: FT } = await import('../src/core/types.js');

    const solver = new Solver2D({
      gravity: { x: 0, y: -9.81 }, dt: 1 / 60, iterations: 10,
    });
    const ground = solver.bodyStore.addBody(Desc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, Coll2D.cuboid(20, 0.5).setFriction(0.5));
    const box = solver.bodyStore.addBody(
      Desc2D.dynamic().setTranslation(0, 1.1).setLinvel(10, 0)
    );
    solver.bodyStore.attachCollider(box.index, Coll2D.cuboid(0.5, 0.5).setFriction(0.5));

    for (let i = 0; i < 30; i++) solver.step();

    const rows = solver.constraintStore.rows;
    for (let i = 0; i + 1 < rows.length; i++) {
      const nRow = rows[i];
      const fRow = rows[i + 1];
      if (!nRow.active || nRow.type !== FT.Contact || isFinite(nRow.fmin)) continue;
      if (!fRow.active || fRow.type !== FT.Contact || !isFinite(fRow.fmin)) continue;
      const mu = 0.5;
      const nForce = Math.abs(nRow.lambda);
      const fForce = Math.abs(fRow.lambda);
      // Coulomb cone: |f_friction| <= mu * |f_normal| (with tolerance)
      expect(fForce).toBeLessThanOrEqual(mu * nForce + 1.0);
      i++;
    }
  });

  it('friction monotonicity: higher friction → less final velocity (2D)', () => {
    const frictions = [0.1, 0.3, 0.5, 0.7, 1.0];
    const finalSpeeds: number[] = [];

    for (const f of frictions) {
      const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
      world.createCollider(ColliderDesc2D.cuboid(20, 0.5).setFriction(f));
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 1.1).setLinvel(5, 0)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3).setFriction(f), body);
      for (let i = 0; i < 90; i++) world.stepCPU();
      finalSpeeds.push(Math.abs(body.linvel().x));
    }

    // Each higher friction should produce same or lower speed
    for (let i = 1; i < finalSpeeds.length; i++) {
      expect(finalSpeeds[i]).toBeLessThanOrEqual(finalSpeeds[i - 1] + 1.0);
    }
  });
});
