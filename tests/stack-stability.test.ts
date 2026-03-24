/**
 * Stack and pyramid stability tests.
 * These verify that multi-body stacking scenarios remain stable over time.
 */
import { describe, it, expect } from 'vitest';
import { World, RigidBodyDesc2D, ColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── 2D Stack Stability ────────────────────────────────────────────────────

describe('2D Stack stability', () => {
  it('5-box stack settles without exploding', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 15 });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5));

    const boxes: any[] = [];
    for (let i = 0; i < 5; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 1.1 + i * 1.05)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5), body);
      boxes.push(body);
    }

    // Simulate 5 seconds
    for (let i = 0; i < 300; i++) world.stepCPU();

    // All boxes above ground
    for (const box of boxes) {
      expect(box.translation().y).toBeGreaterThan(0.3);
      expect(isFinite(box.translation().y)).toBe(true);
    }

    // Boxes should be ordered by height
    for (let i = 1; i < boxes.length; i++) {
      expect(boxes[i].translation().y).toBeGreaterThan(boxes[i - 1].translation().y - 0.5);
    }

    // Horizontal spread should be bounded (not exploding)
    for (const box of boxes) {
      expect(Math.abs(box.translation().x)).toBeLessThan(5);
    }
  });

  it('two boxes side by side settle without NaN', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 15 });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5));

    const a = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(-0.6, 1.1)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5), a);

    const b = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0.6, 1.1)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5), b);

    for (let i = 0; i < 180; i++) {
      world.stepCPU();
      expect(isFinite(a.translation().y)).toBe(true);
      expect(isFinite(b.translation().y)).toBe(true);
    }
  });

  it('many boxes pile without NaN', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5));

    for (let i = 0; i < 15; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation((Math.random() - 0.5) * 2, 2 + i * 0.8)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3), body);
    }

    for (let i = 0; i < 180; i++) {
      world.stepCPU();
      const states = world.getBodyStates();
      for (let j = 0; j < states.length; j++) {
        expect(isFinite(states[j])).toBe(true);
      }
    }
  });
});

// ─── 3D Stack Stability ────────────────────────────────────────────────────

describe('3D Stack stability', () => {
  it('3-box stack settles stably', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true, iterations: 15 });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

    const boxes: any[] = [];
    for (let i = 0; i < 3; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 1.5 + i * 1.1, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), body);
      boxes.push(body);
    }

    for (let i = 0; i < 300; i++) world.step();

    // All above ground
    for (const box of boxes) {
      expect(box.translation().y).toBeGreaterThan(0.3);
      expect(isFinite(box.translation().y)).toBe(true);
    }

    // Ordered vertically
    for (let i = 1; i < boxes.length; i++) {
      expect(boxes[i].translation().y).toBeGreaterThan(boxes[i - 1].translation().y - 0.5);
    }
  });

  it('5-box stack does not explode', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true, iterations: 15 });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

    const boxes: any[] = [];
    for (let i = 0; i < 5; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 1.5 + i * 1.1, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), body);
      boxes.push(body);
    }

    for (let i = 0; i < 300; i++) world.step();

    // All above ground and not launched into space
    for (const box of boxes) {
      const y = box.translation().y;
      expect(y).toBeGreaterThan(0.0);
      expect(y).toBeLessThan(20); // Not exploding upward
      expect(isFinite(y)).toBe(true);
      // Horizontal spread limited
      expect(Math.abs(box.translation().x)).toBeLessThan(5);
      expect(Math.abs(box.translation().z)).toBeLessThan(5);
    }
  });

  it('3D pyramid stays intact', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true, iterations: 15 });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

    const allBoxes: any[] = [];
    // 2x2 bottom, 1x1 top
    for (let x = 0; x < 2; x++) {
      for (let z = 0; z < 2; z++) {
        const body = world.createRigidBody(
          RigidBodyDesc3D.dynamic().setTranslation(x * 1.05 - 0.525, 1.5, z * 1.05 - 0.525)
        );
        world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), body);
        allBoxes.push(body);
      }
    }
    // Top
    const top = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 2.6, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), top);
    allBoxes.push(top);

    for (let i = 0; i < 300; i++) world.step();

    for (const box of allBoxes) {
      expect(box.translation().y).toBeGreaterThan(0.0);
      expect(isFinite(box.translation().y)).toBe(true);
    }
  });
});
