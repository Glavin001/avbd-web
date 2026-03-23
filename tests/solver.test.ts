import { describe, it, expect } from 'vitest';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';

describe('AVBDSolver2D', () => {
  describe('Free fall', () => {
    it('should accelerate a body under gravity', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
        iterations: 10,
      });

      const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 5);
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const body = solver.bodyStore.getBody(handle);
      const initialY = body.position.y;

      // Step a few times
      for (let i = 0; i < 10; i++) {
        solver.step();
      }

      // Body should have fallen
      expect(body.position.y).toBeLessThan(initialY);
      expect(body.velocity.y).toBeLessThan(0);
    });

    it('should maintain horizontal position with no horizontal forces', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
      });

      const desc = RigidBodyDesc2D.dynamic().setTranslation(3, 5);
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const body = solver.bodyStore.getBody(handle);

      for (let i = 0; i < 10; i++) {
        solver.step();
      }

      expect(body.position.x).toBeCloseTo(3, 5);
    });

    it('should match analytical free fall (s = 0.5*g*t^2)', () => {
      const dt = 1 / 60;
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt,
        iterations: 10,
      });

      const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 100);
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const body = solver.bodyStore.getBody(handle);

      const steps = 60; // 1 second
      for (let i = 0; i < steps; i++) {
        solver.step();
      }

      const t = steps * dt;
      const expected = 100 - 0.5 * 9.81 * t * t;
      // Should be close to analytical result (implicit Euler has slight damping)
      expect(body.position.y).toBeCloseTo(expected, 0);
    });
  });

  describe('Fixed bodies', () => {
    it('should not move fixed bodies', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
      });

      const desc = RigidBodyDesc2D.fixed().setTranslation(0, 0);
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(5, 0.5));

      const body = solver.bodyStore.getBody(handle);

      for (let i = 0; i < 10; i++) {
        solver.step();
      }

      expect(body.position.x).toBe(0);
      expect(body.position.y).toBe(0);
      expect(body.angle).toBe(0);
    });
  });

  describe('Contact collision', () => {
    it('should prevent a box from falling through a floor', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
        iterations: 10,
        beta: 100000,
      });

      // Ground (fixed)
      const groundDesc = RigidBodyDesc2D.fixed().setTranslation(0, 0);
      const groundHandle = solver.bodyStore.addBody(groundDesc);
      solver.bodyStore.attachCollider(groundHandle.index, ColliderDesc2D.cuboid(10, 0.5));

      // Falling box
      const boxDesc = RigidBodyDesc2D.dynamic().setTranslation(0, 2);
      const boxHandle = solver.bodyStore.addBody(boxDesc);
      solver.bodyStore.attachCollider(boxHandle.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const box = solver.bodyStore.getBody(boxHandle);

      // Simulate 2 seconds
      for (let i = 0; i < 120; i++) {
        solver.step();
      }

      // Box should rest on ground (ground top = 0.5, box bottom = box.y - 0.5)
      // So box.y should be approximately 1.0
      expect(box.position.y).toBeGreaterThan(0.4);
      expect(box.position.y).toBeLessThan(2.0);
    });

    it('should handle box stacking', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
        iterations: 15,
        beta: 100000,
      });

      // Ground
      const groundDesc = RigidBodyDesc2D.fixed().setTranslation(0, 0);
      const groundHandle = solver.bodyStore.addBody(groundDesc);
      solver.bodyStore.attachCollider(groundHandle.index, ColliderDesc2D.cuboid(10, 0.5));

      // Stack 3 boxes
      const boxes = [];
      for (let i = 0; i < 3; i++) {
        const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 1.5 + i * 1.1);
        const handle = solver.bodyStore.addBody(desc);
        solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));
        boxes.push(solver.bodyStore.getBody(handle));
      }

      // Simulate 3 seconds
      for (let i = 0; i < 180; i++) {
        solver.step();
      }

      // All boxes should be above the ground
      for (const box of boxes) {
        expect(box.position.y).toBeGreaterThan(0.4);
      }

      // Boxes should be ordered (each higher than the one below)
      expect(boxes[1].position.y).toBeGreaterThan(boxes[0].position.y);
      expect(boxes[2].position.y).toBeGreaterThan(boxes[1].position.y);
    });
  });

  describe('Circle physics', () => {
    it('should handle circle-circle collision', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: 0 },
        dt: 1 / 60,
        iterations: 10,
      });

      // Two circles heading towards each other
      const descA = RigidBodyDesc2D.dynamic()
        .setTranslation(-2, 0)
        .setLinvel(5, 0);
      const handleA = solver.bodyStore.addBody(descA);
      solver.bodyStore.attachCollider(handleA.index, ColliderDesc2D.ball(1));

      const descB = RigidBodyDesc2D.dynamic()
        .setTranslation(2, 0)
        .setLinvel(-5, 0);
      const handleB = solver.bodyStore.addBody(descB);
      solver.bodyStore.attachCollider(handleB.index, ColliderDesc2D.ball(1));

      const a = solver.bodyStore.getBody(handleA);
      const b = solver.bodyStore.getBody(handleB);

      for (let i = 0; i < 60; i++) {
        solver.step();
      }

      // Circles should not be overlapping
      const dist = Math.sqrt(
        (a.position.x - b.position.x) ** 2 +
        (a.position.y - b.position.y) ** 2
      );
      expect(dist).toBeGreaterThanOrEqual(1.8); // Combined radius = 2, allow some tolerance
    });
  });

  describe('Initial velocity', () => {
    it('should handle initial horizontal velocity', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
      });

      const desc = RigidBodyDesc2D.dynamic()
        .setTranslation(0, 10)
        .setLinvel(5, 0);
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const body = solver.bodyStore.getBody(handle);

      for (let i = 0; i < 30; i++) {
        solver.step();
      }

      // Should have moved right and fallen
      expect(body.position.x).toBeGreaterThan(0);
      expect(body.position.y).toBeLessThan(10);
    });

    it('should handle initial angular velocity', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: 0 }, // No gravity for clean test
        dt: 1 / 60,
      });

      const desc = RigidBodyDesc2D.dynamic()
        .setTranslation(0, 0)
        .setAngvel(Math.PI); // 180 deg/s
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const body = solver.bodyStore.getBody(handle);

      for (let i = 0; i < 30; i++) {
        solver.step();
      }

      // Should have rotated (0.5 seconds at PI rad/s = PI/2 radians)
      expect(Math.abs(body.angle)).toBeGreaterThan(0);
    });
  });

  describe('Damping', () => {
    it('should reduce velocity with linear damping', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: 0 },
        dt: 1 / 60,
      });

      const desc = RigidBodyDesc2D.dynamic()
        .setTranslation(0, 0)
        .setLinvel(10, 0)
        .setLinearDamping(5);
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const body = solver.bodyStore.getBody(handle);

      for (let i = 0; i < 60; i++) {
        solver.step();
      }

      // Velocity should have decreased significantly
      expect(Math.abs(body.velocity.x)).toBeLessThan(5);
    });
  });

  describe('Energy conservation (no gravity)', () => {
    it('should approximately conserve energy in elastic collision', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: 0 },
        dt: 1 / 60,
        iterations: 20,
      });

      const descA = RigidBodyDesc2D.dynamic()
        .setTranslation(-3, 0)
        .setLinvel(5, 0);
      const handleA = solver.bodyStore.addBody(descA);
      solver.bodyStore.attachCollider(handleA.index, ColliderDesc2D.ball(1).setRestitution(1));

      const descB = RigidBodyDesc2D.dynamic()
        .setTranslation(3, 0)
        .setLinvel(-5, 0);
      const handleB = solver.bodyStore.addBody(descB);
      solver.bodyStore.attachCollider(handleB.index, ColliderDesc2D.ball(1).setRestitution(1));

      const a = solver.bodyStore.getBody(handleA);
      const b = solver.bodyStore.getBody(handleB);

      const initialKE = 0.5 * a.mass * (5 * 5) + 0.5 * b.mass * (5 * 5);

      for (let i = 0; i < 120; i++) {
        solver.step();
      }

      const finalVA = Math.sqrt(a.velocity.x ** 2 + a.velocity.y ** 2);
      const finalVB = Math.sqrt(b.velocity.x ** 2 + b.velocity.y ** 2);
      const finalKE = 0.5 * a.mass * finalVA ** 2 + 0.5 * b.mass * finalVB ** 2;

      // AVBD is an implicit solver, so some energy loss is expected
      // but it shouldn't gain energy
      expect(finalKE).toBeLessThanOrEqual(initialKE * 1.1);
    });
  });

  describe('Solver parameters', () => {
    it('should use default parameters', () => {
      const solver = new AVBDSolver2D();
      expect(solver.config.iterations).toBe(10);
      expect(solver.config.dt).toBeCloseTo(1 / 60);
      expect(solver.config.beta).toBe(100000);
      expect(solver.config.alpha).toBe(0.99);
      expect(solver.config.gamma).toBe(0.99);
      expect(solver.config.postStabilize).toBe(true);
    });

    it('should accept custom parameters', () => {
      const solver = new AVBDSolver2D({
        iterations: 20,
        beta: 50000,
        postStabilize: false,
      });
      expect(solver.config.iterations).toBe(20);
      expect(solver.config.beta).toBe(50000);
      expect(solver.config.postStabilize).toBe(false);
    });
  });

  describe('Multiple bodies without collision', () => {
    it('should simulate independent bodies correctly', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
      });

      // Two bodies far apart, should not interact
      const descA = RigidBodyDesc2D.dynamic().setTranslation(-10, 5);
      const handleA = solver.bodyStore.addBody(descA);
      solver.bodyStore.attachCollider(handleA.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const descB = RigidBodyDesc2D.dynamic().setTranslation(10, 5);
      const handleB = solver.bodyStore.addBody(descB);
      solver.bodyStore.attachCollider(handleB.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const a = solver.bodyStore.getBody(handleA);
      const b = solver.bodyStore.getBody(handleB);

      for (let i = 0; i < 30; i++) {
        solver.step();
      }

      // Both should have the same y position (symmetric)
      expect(a.position.y).toBeCloseTo(b.position.y, 5);
      // X positions should be unchanged
      expect(a.position.x).toBeCloseTo(-10, 5);
      expect(b.position.x).toBeCloseTo(10, 5);
    });
  });

  describe('Gravity scale', () => {
    it('should respect gravity scale', () => {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
      });

      const descA = RigidBodyDesc2D.dynamic().setTranslation(-5, 10);
      const handleA = solver.bodyStore.addBody(descA);
      solver.bodyStore.attachCollider(handleA.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const descB = RigidBodyDesc2D.dynamic().setTranslation(5, 10).setGravityScale(0);
      const handleB = solver.bodyStore.addBody(descB);
      solver.bodyStore.attachCollider(handleB.index, ColliderDesc2D.cuboid(0.5, 0.5));

      const a = solver.bodyStore.getBody(handleA);
      const b = solver.bodyStore.getBody(handleB);

      for (let i = 0; i < 30; i++) {
        solver.step();
      }

      // A should have fallen, B should stay at same height
      expect(a.position.y).toBeLessThan(10);
      expect(b.position.y).toBeCloseTo(10, 3);
    });
  });
});
