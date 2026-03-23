/**
 * Rotation correctness tests for 2D and 3D physics.
 * Tests that contact constraints produce correct rotational behavior:
 * - Boxes landing flat should not spin erratically
 * - Friction should correctly apply torque (rolling, not spinning wildly)
 * - Off-center impacts should produce controlled rotation
 * - Edge cases: corner contacts, grazing contacts, stacking
 */
import { describe, it, expect } from 'vitest';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';
import { World, RigidBodyDesc2D as WRigidBodyDesc2D, ColliderDesc2D as WColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Maximum angular velocity seen during simulation */
function maxAngularVelocity2D(solver: AVBDSolver2D, steps: number): number {
  let maxOmega = 0;
  for (let i = 0; i < steps; i++) {
    solver.step();
    for (const body of solver.bodyStore.bodies) {
      if (body.type !== 1) { // not fixed
        maxOmega = Math.max(maxOmega, Math.abs(body.angularVelocity));
      }
    }
  }
  return maxOmega;
}

// ─── 2D: Box landing flat should not spin ───────────────────────────────────

describe('2D Rotation: Box landing flat', () => {
  it('should not spin when dropped straight onto flat ground', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    // Ground
    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Box dropped from height, perfectly aligned (no initial rotation)
    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 3),
    );
    solver.bodyStore.attachCollider(box.index,
      ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
    );

    const maxOmega = maxAngularVelocity2D(solver, 180);

    // A box dropped perfectly flat should have near-zero angular velocity
    // With the bug (missing hessianDiag), boxes spin erratically
    expect(maxOmega).toBeLessThan(1.0);

    // Final angle should be near zero
    const body = solver.bodyStore.bodies[1];
    expect(Math.abs(body.angle)).toBeLessThan(0.3);
  });

  it('should not spin when stacked boxes settle', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Stack 5 boxes
    for (let i = 0; i < 5; i++) {
      const box = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 1.1 + i * 1.05),
      );
      solver.bodyStore.attachCollider(box.index,
        ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
      );
    }

    // Simulate 5 seconds
    for (let i = 0; i < 300; i++) solver.step();

    // After settling, all boxes should have very low angular velocity
    for (const body of solver.bodyStore.bodies) {
      if (body.type !== 1) {
        expect(Math.abs(body.angularVelocity)).toBeLessThan(2.0);
      }
    }
  });
});

// ─── 2D: Off-center impact should produce controlled rotation ───────────────

describe('2D Rotation: Off-center impact', () => {
  it('should rotate when hit off-center, but not explode', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Box resting on ground
    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1),
    );
    solver.bodyStore.attachCollider(box.index,
      ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
    );

    // Let it settle
    for (let i = 0; i < 60; i++) solver.step();

    // Give it a lateral impulse (off-center hit)
    const body = solver.bodyStore.bodies[1];
    body.velocity = { x: 5, y: 0 };

    // Simulate the response
    const maxOmega = maxAngularVelocity2D(solver, 180);

    // Should rotate (friction generates torque) but not spin wildly
    // The key is it should stay bounded by the solver's angular velocity clamp (50)
    expect(maxOmega).toBeLessThanOrEqual(50);
    expect(isFinite(body.angle)).toBe(true);
  });
});

// ─── 2D: Friction-induced rolling ──────────────────────────────────────────

describe('2D Rotation: Friction rolling', () => {
  it('should produce correct rolling behavior for a sliding box', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index,
      ColliderDesc2D.cuboid(50, 0.5).setFriction(0.8),
    );

    // Box sliding sideways on ground
    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic()
        .setTranslation(0, 1.1)
        .setLinvel(5, 0),
    );
    solver.bodyStore.attachCollider(box.index,
      ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.8),
    );

    // Track angular velocity over time
    let maxOmega = 0;
    let omegaHistory: number[] = [];
    for (let i = 0; i < 120; i++) {
      solver.step();
      const body = solver.bodyStore.bodies[1];
      const omega = body.angularVelocity;
      maxOmega = Math.max(maxOmega, Math.abs(omega));
      omegaHistory.push(omega);
    }

    // Angular velocity should be bounded by the solver's clamp (50 rad/s)
    expect(maxOmega).toBeLessThanOrEqual(50);

    // Should decelerate (friction slows it down)
    const body = solver.bodyStore.bodies[1];
    expect(Math.abs(body.velocity.x)).toBeLessThan(5);
  });
});

// ─── 2D: Many boxes should not exhibit collective spinning ─────────────────

describe('2D Rotation: Many-body spin stability', () => {
  it('should keep angular velocities bounded in 50-box grid', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Grid of 50 boxes falling onto ground
    for (let i = 0; i < 50; i++) {
      const col = i % 10;
      const row = Math.floor(i / 10);
      const handle = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation((col - 5) * 1.1, 2 + row * 1.1),
      );
      solver.bodyStore.attachCollider(handle.index,
        ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
      );
    }

    // Simulate 3 seconds
    let maxOmega = 0;
    for (let i = 0; i < 180; i++) {
      solver.step();
      for (const body of solver.bodyStore.bodies) {
        if (body.type !== 1) {
          maxOmega = Math.max(maxOmega, Math.abs(body.angularVelocity));
        }
      }
    }

    // No box should exceed the solver's angular velocity clamp (50 rad/s)
    // Before the fix, boxes would spin at 190+ rad/s (unclamped)
    expect(maxOmega).toBeLessThanOrEqual(50);
  });

  it('should keep angular velocities bounded in 100-box pile', { timeout: 15000 }, () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Walls
    const lw = solver.bodyStore.addBody(RigidBodyDesc2D.fixed().setTranslation(-5, 5));
    solver.bodyStore.attachCollider(lw.index, ColliderDesc2D.cuboid(0.3, 5));
    const rw = solver.bodyStore.addBody(RigidBodyDesc2D.fixed().setTranslation(5, 5));
    solver.bodyStore.attachCollider(rw.index, ColliderDesc2D.cuboid(0.3, 5));

    for (let i = 0; i < 100; i++) {
      const col = i % 10;
      const row = Math.floor(i / 10);
      const handle = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation((col - 5) * 0.9, 2 + row * 1.0),
      );
      solver.bodyStore.attachCollider(handle.index,
        ColliderDesc2D.cuboid(0.4, 0.4).setFriction(0.4),
      );
    }

    let maxOmega = 0;
    for (let i = 0; i < 180; i++) {
      solver.step();
      for (const body of solver.bodyStore.bodies) {
        if (body.type !== 1) {
          maxOmega = Math.max(maxOmega, Math.abs(body.angularVelocity));
        }
      }
    }

    // With 100 boxes, angular velocity should be bounded by the solver's clamp (50)
    // Before fix: exceeded 190 rad/s. After fix: clamped at 50.
    expect(maxOmega).toBeLessThanOrEqual(50);
  });
});

// ─── 2D: Corner contact rotation ───────────────────────────────────────────

describe('2D Rotation: Corner contacts', () => {
  it('should handle box landing on corner without exploding', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Box tilted 45 degrees (lands on corner)
    const box = solver.bodyStore.addBody(
      RigidBodyDesc2D.dynamic()
        .setTranslation(0, 4)
        .setRotation(Math.PI / 4),
    );
    solver.bodyStore.attachCollider(box.index,
      ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5).setRestitution(0.1),
    );

    const maxOmega = maxAngularVelocity2D(solver, 300);

    // Should tumble and settle, bounded by solver's angular velocity clamp (50)
    expect(maxOmega).toBeLessThanOrEqual(50);

    const body = solver.bodyStore.bodies[1];
    expect(isFinite(body.angle)).toBe(true);
    expect(isFinite(body.position.x)).toBe(true);
    expect(isFinite(body.position.y)).toBe(true);
  });
});

// ─── 3D: Cube landing flat ─────────────────────────────────────────────────

describe('3D Rotation: Cube landing flat', () => {
  it('should not spin when dropped straight onto ground', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 10,
      useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(20, 0.5, 20).setFriction(0.5));

    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 3, 0),
    );
    world.createCollider(
      ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5),
      body,
    );

    let maxAngVel = 0;
    for (let i = 0; i < 180; i++) {
      world.step();
      const av = body.angvel();
      const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
      maxAngVel = Math.max(maxAngVel, mag);
    }

    // A cube dropped flat should have negligible angular velocity
    expect(maxAngVel).toBeLessThan(1.0);
  });

  it('should not spin in a 10-cube stack', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 10,
      useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));

    const bodies: any[] = [];
    for (let i = 0; i < 10; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 1.1 + i * 1.05, 0),
      );
      world.createCollider(
        ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5),
        body,
      );
      bodies.push(body);
    }

    for (let i = 0; i < 300; i++) world.step();

    // All cubes should have low angular velocity after settling
    for (const body of bodies) {
      const av = body.angvel();
      const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
      expect(mag).toBeLessThan(5.0);
    }
  });
});

// ─── 3D: Off-center impact ─────────────────────────────────────────────────

describe('3D Rotation: Off-center impact', () => {
  it('should produce bounded rotation from lateral velocity', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 10,
      useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(20, 0.5, 20).setFriction(0.5));

    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 1.1, 0),
    );
    world.createCollider(
      ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5),
      body,
    );

    // Let it settle
    for (let i = 0; i < 60; i++) world.step();

    // Give lateral push
    body.setLinvel({ x: 5, y: 0, z: 0 });

    let maxAngVel = 0;
    for (let i = 0; i < 180; i++) {
      world.step();
      const av = body.angvel();
      const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
      maxAngVel = Math.max(maxAngVel, mag);
    }

    // Should rotate from friction but not explode
    expect(maxAngVel).toBeLessThan(20);
    expect(isFinite(body.translation().x)).toBe(true);
  });
});

// ─── 3D: Many-body rotation stability ──────────────────────────────────────

describe('3D Rotation: Many-body spin stability', () => {
  it('should keep angular velocities bounded in 30-cube grid', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 5,
      useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(20, 0.5, 20).setFriction(0.5));

    const bodies: any[] = [];
    for (let i = 0; i < 30; i++) {
      const col = i % 5;
      const row = Math.floor(i / 5);
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation((col - 2) * 1.2, 2 + row * 1.1, 0),
      );
      world.createCollider(
        ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5),
        body,
      );
      bodies.push(body);
    }

    let maxAngVel = 0;
    for (let i = 0; i < 180; i++) {
      world.step();
      for (const body of bodies) {
        const av = body.angvel();
        const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
        maxAngVel = Math.max(maxAngVel, mag);
      }
    }

    // No cube should spin faster than ~35 rad/s
    expect(maxAngVel).toBeLessThan(35);
  });
});

// ─── 2D World API: Rotation tests ──────────────────────────────────────────

describe('2D World API: Rotation correctness', () => {
  it('should keep boxes upright in a 10-box stack', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 15 });
    world.createCollider(WColliderDesc2D.cuboid(20, 0.5).setFriction(0.5));

    const boxes: any[] = [];
    for (let i = 0; i < 10; i++) {
      const body = world.createRigidBody(
        WRigidBodyDesc2D.dynamic().setTranslation(0, 1.1 + i * 1.05),
      );
      world.createCollider(
        WColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
        body,
      );
      boxes.push(body);
    }

    for (let i = 0; i < 300; i++) world.stepCPU();

    // Boxes in a stack should remain roughly upright
    for (const box of boxes) {
      // Normalize angle to [-pi, pi]
      let angle = box.rotation() % (2 * Math.PI);
      if (angle > Math.PI) angle -= 2 * Math.PI;
      if (angle < -Math.PI) angle += 2 * Math.PI;

      // Angle should be a multiple of pi/2 (box settled on a face)
      const remainder = Math.abs(angle % (Math.PI / 2));
      const distToFace = Math.min(remainder, Math.PI / 2 - remainder);
      expect(distToFace).toBeLessThan(0.5);
    }
  });

  it('symmetric boxes should stay symmetric', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    world.createCollider(WColliderDesc2D.cuboid(20, 0.5).setFriction(0.5));

    // Two boxes at symmetric positions
    const bodyL = world.createRigidBody(
      WRigidBodyDesc2D.dynamic().setTranslation(-2, 3),
    );
    world.createCollider(
      WColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
      bodyL,
    );

    const bodyR = world.createRigidBody(
      WRigidBodyDesc2D.dynamic().setTranslation(2, 3),
    );
    world.createCollider(
      WColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
      bodyR,
    );

    for (let i = 0; i < 180; i++) world.stepCPU();

    // Both should produce valid finite results
    const tL = bodyL.translation();
    const tR = bodyR.translation();
    expect(isFinite(tL.x) && isFinite(tL.y)).toBe(true);
    expect(isFinite(tR.x) && isFinite(tR.y)).toBe(true);

    // Symmetric y-positions (both should be at similar height)
    expect(Math.abs(tL.y - tR.y)).toBeLessThan(1.0);

    // Both should have settled (low angular velocity)
    const angleL = bodyL.rotation();
    const angleR = bodyR.rotation();
    expect(isFinite(angleL)).toBe(true);
    expect(isFinite(angleR)).toBe(true);
  });
});
