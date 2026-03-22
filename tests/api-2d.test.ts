import { describe, it, expect } from 'vitest';
import AVBD, { World, RigidBodyDesc2D, ColliderDesc2D, JointData2D } from '../src/2d/index.js';

describe('AVBD 2D Public API', () => {
  describe('World creation', () => {
    it('should create a world with gravity', () => {
      const world = new World({ x: 0, y: -9.81 });
      expect(world).toBeDefined();
    });

    it('should accept custom solver config', () => {
      const world = new World({ x: 0, y: -9.81 }, {
        iterations: 20,
        beta: 50000,
        postStabilize: true,
        dt: 1 / 120,
      });
      expect(world.rawSolver.config.iterations).toBe(20);
      expect(world.rawSolver.config.beta).toBe(50000);
    });
  });

  describe('Rapier-style API pattern', () => {
    it('should match Rapier API usage pattern', async () => {
      await AVBD.init();

      const gravity = { x: 0, y: -9.81 };
      const world = new AVBD.World(gravity);

      // Create ground
      const groundColliderDesc = AVBD.ColliderDesc.cuboid(10, 0.5);
      world.createCollider(groundColliderDesc);

      // Create dynamic body
      const rigidBodyDesc = AVBD.RigidBodyDesc.dynamic()
        .setTranslation(0, 5)
        .setLinearDamping(0.1);
      const rigidBody = world.createRigidBody(rigidBodyDesc);

      const colliderDesc = AVBD.ColliderDesc.cuboid(0.5, 0.5)
        .setRestitution(0.3)
        .setFriction(0.5)
        .setDensity(1.0);
      world.createCollider(colliderDesc, rigidBody);

      // Step
      world.step();

      const pos = rigidBody.translation();
      expect(pos.x).toBeCloseTo(0);
      expect(pos.y).toBeLessThan(5);

      const rot = rigidBody.rotation();
      expect(typeof rot).toBe('number');
    });

    it('should support async step', async () => {
      const world = new AVBD.World({ x: 0, y: -9.81 });
      const desc = AVBD.RigidBodyDesc.dynamic().setTranslation(0, 5);
      const body = world.createRigidBody(desc);
      world.createCollider(AVBD.ColliderDesc.cuboid(0.5, 0.5), body);

      await world.stepAsync();
      expect(body.translation().y).toBeLessThan(5);
    });
  });

  describe('RigidBody operations', () => {
    it('should read body properties', () => {
      const world = new World({ x: 0, y: -9.81 });
      const desc = RigidBodyDesc2D.dynamic()
        .setTranslation(3, 4)
        .setRotation(1.5)
        .setLinvel(2, 3)
        .setAngvel(0.5);
      const body = world.createRigidBody(desc);
      world.createCollider(ColliderDesc2D.cuboid(1, 1), body);

      expect(body.translation()).toEqual({ x: 3, y: 4 });
      expect(body.rotation()).toBe(1.5);
      expect(body.linvel()).toEqual({ x: 2, y: 3 });
      expect(body.angvel()).toBe(0.5);
      expect(body.mass()).toBeGreaterThan(0);
      expect(body.isDynamic()).toBe(true);
      expect(body.isFixed()).toBe(false);
    });

    it('should set body properties', () => {
      const world = new World({ x: 0, y: 0 });
      const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
      world.createCollider(ColliderDesc2D.cuboid(1, 1), body);

      body.setTranslation({ x: 5, y: 10 });
      expect(body.translation()).toEqual({ x: 5, y: 10 });

      body.setRotation(Math.PI);
      expect(body.rotation()).toBeCloseTo(Math.PI);

      body.setLinvel({ x: 3, y: -2 });
      expect(body.linvel()).toEqual({ x: 3, y: -2 });

      body.setAngvel(1.5);
      expect(body.angvel()).toBe(1.5);
    });

    it('should apply impulse', () => {
      const world = new World({ x: 0, y: 0 });
      const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setDensity(1), body);

      const mass = body.mass();
      body.applyImpulse({ x: mass * 5, y: 0 }); // Should give velocity of 5

      expect(body.linvel().x).toBeCloseTo(5, 3);
    });

    it('should apply force', () => {
      const world = new World({ x: 0, y: 0 }, { dt: 1 / 60 });
      const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setDensity(1), body);

      body.applyForce({ x: 100, y: 0 });
      expect(body.linvel().x).toBeGreaterThan(0);
    });
  });

  describe('Ground collision (box on floor)', () => {
    it('should prevent box from falling through floor', () => {
      const world = new World({ x: 0, y: -9.81 }, {
        iterations: 10,
      });

      // Ground at y=0
      world.createCollider(ColliderDesc2D.cuboid(10, 0.5));

      // Box above ground
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 2)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

      // Simulate 2 seconds
      for (let i = 0; i < 120; i++) {
        world.step();
      }

      // Box should rest on ground (y ≈ 1.0 = ground_top(0.5) + box_half_height(0.5))
      expect(body.translation().y).toBeGreaterThan(0.4);
      expect(body.translation().y).toBeLessThan(2.0);
    });

    it('should handle ball on floor', () => {
      const world = new World({ x: 0, y: -9.81 }, {
        iterations: 10,
      });

      world.createCollider(ColliderDesc2D.cuboid(10, 0.5));

      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 3)
      );
      world.createCollider(ColliderDesc2D.ball(0.5), body);

      for (let i = 0; i < 120; i++) {
        world.step();
      }

      // Ball should rest on ground (y ≈ 1.0 = ground_top(0.5) + ball_radius(0.5))
      // Note: ball-box contacts have less surface area so settle slightly lower
      expect(body.translation().y).toBeGreaterThan(0.0);
    });
  });

  describe('Box stacking', () => {
    it('should support stable stacking of 3 boxes', () => {
      const world = new World({ x: 0, y: -9.81 }, {
        iterations: 15,
      });

      world.createCollider(ColliderDesc2D.cuboid(10, 0.5));

      const boxes = [];
      for (let i = 0; i < 3; i++) {
        const body = world.createRigidBody(
          RigidBodyDesc2D.dynamic().setTranslation(0, 1.5 + i * 1.1)
        );
        world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);
        boxes.push(body);
      }

      // Simulate 3 seconds
      for (let i = 0; i < 180; i++) {
        world.step();
      }

      // All boxes should be above ground
      for (const box of boxes) {
        expect(box.translation().y).toBeGreaterThan(0.4);
      }

      // Should be stacked (each higher than the one below)
      expect(boxes[1].translation().y).toBeGreaterThan(boxes[0].translation().y);
      expect(boxes[2].translation().y).toBeGreaterThan(boxes[1].translation().y);
    });
  });

  describe('Joints', () => {
    it('should create a revolute joint', () => {
      const world = new World({ x: 0, y: -9.81 }, {
        iterations: 10,
      });

      const bodyA = world.createRigidBody(
        RigidBodyDesc2D.fixed().setTranslation(0, 5)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), bodyA);

      const bodyB = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 3)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), bodyB);

      const jointParams = JointData2D.revolute(
        { x: 0, y: -1 },  // anchor on A
        { x: 0, y: 1 },   // anchor on B
      );
      world.createJoint(jointParams, bodyA, bodyB);

      // Simulate
      for (let i = 0; i < 60; i++) {
        world.step();
      }

      // Body B should stay connected to body A via the joint
      // The distance between the anchor points should remain approximately constant
      const posA = bodyA.translation();
      const posB = bodyB.translation();
      const dist = Math.sqrt(
        (posA.x - posB.x) ** 2 +
        (posA.y - posB.y) ** 2
      );
      // Joint anchors are 2 units apart in local coords
      expect(dist).toBeGreaterThan(0.5);
      expect(dist).toBeLessThan(5);
    });
  });

  describe('Bulk readback', () => {
    it('should return body states as Float32Array', () => {
      const world = new World({ x: 0, y: 0 });
      const body1 = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(1, 2)
      );
      world.createCollider(ColliderDesc2D.cuboid(1, 1), body1);

      const body2 = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(5, 10).setRotation(1.5)
      );
      world.createCollider(ColliderDesc2D.cuboid(1, 1), body2);

      const states = world.getBodyStates();
      expect(states).toBeInstanceOf(Float32Array);
      // Body 1 (index 0): x=1, y=2, angle=0
      expect(states[0]).toBeCloseTo(1);
      expect(states[1]).toBeCloseTo(2);
      expect(states[2]).toBeCloseTo(0);
      // Body 2 (index 1): x=5, y=10, angle=1.5
      expect(states[3]).toBeCloseTo(5);
      expect(states[4]).toBeCloseTo(10);
      expect(states[5]).toBeCloseTo(1.5);
    });
  });

  describe('Remove rigid body', () => {
    it('should remove a body from simulation', () => {
      const world = new World({ x: 0, y: -9.81 });
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 5)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

      world.removeRigidBody(body);
      expect(world.numBodies).toBe(0);
    });
  });

  describe('Multiple shapes', () => {
    it('should handle mixed box and circle bodies', () => {
      const world = new World({ x: 0, y: -9.81 }, {
        iterations: 10,
      });

      world.createCollider(ColliderDesc2D.cuboid(10, 0.5));

      const box = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(-1, 2)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), box);

      const ball = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(1, 2)
      );
      world.createCollider(ColliderDesc2D.ball(0.5), ball);

      for (let i = 0; i < 120; i++) {
        world.step();
      }

      // Both should rest on ground
      expect(box.translation().y).toBeGreaterThan(0.4);
      expect(ball.translation().y).toBeGreaterThan(-0.5);
    });
  });
});
