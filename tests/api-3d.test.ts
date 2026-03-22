import { describe, it, expect } from 'vitest';
import AVBD3D, { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

describe('AVBD 3D Public API', () => {
  describe('World creation', () => {
    it('should create a 3D world with gravity', () => {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
      expect(world).toBeDefined();
    });
  });

  describe('Free fall', () => {
    it('should accelerate a body under gravity', () => {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
        iterations: 10, useCPU: true,
      });

      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

      const initialY = body.translation().y;

      for (let i = 0; i < 30; i++) {
        world.step();
      }

      expect(body.translation().y).toBeLessThan(initialY);
      expect(body.linvel().y).toBeLessThan(0);
      // X and Z should be unchanged
      expect(body.translation().x).toBeCloseTo(0, 3);
      expect(body.translation().z).toBeCloseTo(0, 3);
    });

    it('should match analytical free fall approximately', () => {
      const dt = 1 / 60;
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, { dt, useCPU: true });

      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 100, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

      for (let i = 0; i < 60; i++) world.step();

      const t = 60 * dt;
      const expected = 100 - 0.5 * 9.81 * t * t;
      expect(body.translation().y).toBeCloseTo(expected, 0);
    });
  });

  describe('Fixed bodies', () => {
    it('should not move fixed bodies', () => {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });

      const body = world.createRigidBody(
        RigidBodyDesc3D.fixed().setTranslation(1, 2, 3)
      );
      world.createCollider(ColliderDesc3D.cuboid(5, 0.5, 5), body);

      for (let i = 0; i < 30; i++) world.step();

      expect(body.translation()).toEqual({ x: 1, y: 2, z: 3 });
    });
  });

  describe('Box on ground (3D)', () => {
    it('should prevent a cube from falling through a ground plane', () => {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
        iterations: 10, useCPU: true,
      });

      // Ground
      world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

      // Falling cube
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 3, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

      for (let i = 0; i < 120; i++) world.step();

      // Should rest on ground (ground top = 0.5, box half = 0.5, so y ≈ 1.0)
      expect(body.translation().y).toBeGreaterThan(0.3);
      expect(body.translation().y).toBeLessThan(3.0);
    });
  });

  describe('Sphere collision', () => {
    it('should handle sphere-sphere collision', () => {
      const world = new World3D({ x: 0, y: 0, z: 0 }, {
        iterations: 10, useCPU: true,
      });

      const a = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(-2, 0, 0).setLinvel(5, 0, 0)
      );
      world.createCollider(ColliderDesc3D.ball(1), a);

      const b = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(2, 0, 0).setLinvel(-5, 0, 0)
      );
      world.createCollider(ColliderDesc3D.ball(1), b);

      for (let i = 0; i < 60; i++) world.step();

      const dist = Math.sqrt(
        (a.translation().x - b.translation().x) ** 2 +
        (a.translation().y - b.translation().y) ** 2 +
        (a.translation().z - b.translation().z) ** 2
      );
      expect(dist).toBeGreaterThanOrEqual(1.5);
    });
  });

  describe('Body operations', () => {
    it('should read/write body properties', () => {
      const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(1, 2, 3)
      );
      world.createCollider(ColliderDesc3D.cuboid(1, 1, 1), body);

      expect(body.translation()).toEqual({ x: 1, y: 2, z: 3 });
      expect(body.mass()).toBeGreaterThan(0);
      expect(body.isDynamic()).toBe(true);

      body.setTranslation({ x: 5, y: 6, z: 7 });
      expect(body.translation()).toEqual({ x: 5, y: 6, z: 7 });

      body.setLinvel({ x: 1, y: 2, z: 3 });
      expect(body.linvel()).toEqual({ x: 1, y: 2, z: 3 });
    });

    it('should apply impulse', () => {
      const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });
      const body = world.createRigidBody(RigidBodyDesc3D.dynamic());
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setDensity(1), body);

      const mass = body.mass();
      body.applyImpulse({ x: mass * 5, y: 0, z: 0 });
      expect(body.linvel().x).toBeCloseTo(5, 3);
    });

    it('should return quaternion rotation', () => {
      const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });
      const body = world.createRigidBody(RigidBodyDesc3D.dynamic());
      world.createCollider(ColliderDesc3D.cuboid(1, 1, 1), body);

      const rot = body.rotation();
      expect(rot.w).toBeCloseTo(1);
      expect(rot.x).toBeCloseTo(0);
      expect(rot.y).toBeCloseTo(0);
      expect(rot.z).toBeCloseTo(0);
    });
  });

  describe('Bulk readback', () => {
    it('should return body states as Float32Array', () => {
      const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(1, 2, 3)
      );
      world.createCollider(ColliderDesc3D.cuboid(1, 1, 1), body);

      const states = world.getBodyStates();
      expect(states).toBeInstanceOf(Float32Array);
      // [x, y, z, qw, qx, qy, qz]
      expect(states[0]).toBeCloseTo(1);
      expect(states[1]).toBeCloseTo(2);
      expect(states[2]).toBeCloseTo(3);
      expect(states[3]).toBeCloseTo(1); // qw
    });
  });

  describe('Rapier-style API', () => {
    it('should work with AVBD3D namespace (CPU mode for testing)', () => {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
      world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 5, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

      world.step();
      expect(body.translation().y).toBeLessThan(5);
    });

    it('should require AVBD3D.init() for GPU mode', () => {
      expect(() => new World3D({ x: 0, y: -9.81, z: 0 })).toThrow(/AVBD3D.init/);
    });
  });

  describe('Gravity scale', () => {
    it('should respect gravity scale', () => {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });

      const a = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(-5, 10, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), a);

      const b = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(5, 10, 0).setGravityScale(0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), b);

      for (let i = 0; i < 30; i++) world.step();

      expect(a.translation().y).toBeLessThan(10);
      expect(b.translation().y).toBeCloseTo(10, 3);
    });
  });

  describe('Box stacking 3D', () => {
    it('should support stacking 3 cubes', () => {
      const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
        iterations: 15, useCPU: true,
      });

      world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

      const boxes = [];
      for (let i = 0; i < 3; i++) {
        const body = world.createRigidBody(
          RigidBodyDesc3D.dynamic().setTranslation(0, 1.5 + i * 1.1, 0)
        );
        world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);
        boxes.push(body);
      }

      for (let i = 0; i < 180; i++) world.step();

      for (const box of boxes) {
        expect(box.translation().y).toBeGreaterThan(0.3);
      }
      expect(boxes[1].translation().y).toBeGreaterThan(boxes[0].translation().y);
      expect(boxes[2].translation().y).toBeGreaterThan(boxes[1].translation().y);
    });
  });
});
