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
