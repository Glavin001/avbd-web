/**
 * Analytical physics tests — scenarios where the correct answer is mathematically known.
 */
import { describe, it, expect } from 'vitest';
import { World, RigidBodyDesc2D, ColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── 2D Analytical Tests ────────────────────────────────────────────────────

describe('2D Analytical physics', () => {
  it('free fall matches y = y0 - 0.5*g*t^2', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 50)
    );
    world.createCollider(ColliderDesc2D.ball(0.5), body);

    for (let i = 0; i < 60; i++) world.stepCPU();

    const t = 1; // 60 steps at 1/60 = 1 second
    const expected = 50 - 0.5 * 9.81 * t * t; // ≈ 45.095
    expect(body.translation().y).toBeCloseTo(expected, 0); // within 0.5
    // No horizontal drift
    expect(Math.abs(body.translation().x)).toBeLessThan(0.01);
  });

  it('projectile traces parabolic arc (no contacts)', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0).setLinvel(5, 10)
    );
    world.createCollider(ColliderDesc2D.ball(0.3), body);

    for (let i = 0; i < 60; i++) world.stepCPU();

    const t = 1;
    const expectedX = 5 * t; // ≈ 5
    const expectedY = 10 * t - 0.5 * 9.81 * t * t; // ≈ 5.095
    expect(body.translation().x).toBeCloseTo(expectedX, 0);
    expect(body.translation().y).toBeCloseTo(expectedY, 0);
  });

  it('zero-gravity equal-mass head-on collision: momentum conserved', () => {
    const world = new World({ x: 0, y: 0 }, { iterations: 10 });

    const a = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(-3, 0).setLinvel(3, 0)
    );
    world.createCollider(ColliderDesc2D.ball(0.5).setFriction(0).setRestitution(0.5), a);

    const b = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(3, 0).setLinvel(-3, 0)
    );
    world.createCollider(ColliderDesc2D.ball(0.5).setFriction(0).setRestitution(0.5), b);

    const p0 = a.linvel().x + b.linvel().x; // = 0

    for (let i = 0; i < 120; i++) world.stepCPU();

    const p1 = a.linvel().x + b.linvel().x;
    // AVBD is implicit — momentum conservation is approximate.
    // Check it doesn't drift catastrophically.
    expect(Math.abs(p1 - p0)).toBeLessThan(3.0);

    // Ball A should have slowed or reversed after collision
    expect(a.linvel().x).toBeLessThan(3);
  });

  it('stationary body should stay at rest (zero gravity)', () => {
    const world = new World({ x: 0, y: 0 }, { iterations: 10 });
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    for (let i = 0; i < 60; i++) world.stepCPU();

    expect(body.translation().x).toBeCloseTo(0, 2);
    expect(body.translation().y).toBeCloseTo(0, 2);
    expect(body.linvel().x).toBeCloseTo(0, 2);
    expect(body.linvel().y).toBeCloseTo(0, 2);
  });

  it('constant velocity with no gravity or contacts', () => {
    const world = new World({ x: 0, y: 0 }, { iterations: 10 });
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0).setLinvel(3, 4)
    );
    world.createCollider(ColliderDesc2D.ball(0.5), body);

    for (let i = 0; i < 60; i++) world.stepCPU();

    // After 1 second: position = (3, 4)
    expect(body.translation().x).toBeCloseTo(3, 0);
    expect(body.translation().y).toBeCloseTo(4, 0);
  });
});

// ─── 3D Analytical Tests ────────────────────────────────────────────────────

describe('3D Analytical physics', () => {
  it('free fall matches analytical formula', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 50, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    for (let i = 0; i < 60; i++) world.step();

    const expected = 50 - 0.5 * 9.81 * 1;
    expect(body.translation().y).toBeCloseTo(expected, 0);
    // No drift in x or z
    expect(Math.abs(body.translation().x)).toBeLessThan(0.01);
    expect(Math.abs(body.translation().z)).toBeLessThan(0.01);
  });

  it('zero-gravity sphere collision: momentum conserved', () => {
    const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });

    const a = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(-3, 0, 0).setLinvel(3, 0, 0)
    );
    world.createCollider(ColliderDesc3D.ball(0.5), a);

    const b = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(3, 0, 0).setLinvel(-3, 0, 0)
    );
    world.createCollider(ColliderDesc3D.ball(0.5), b);

    const p0x = a.linvel().x + b.linvel().x;
    const p0y = a.linvel().y + b.linvel().y;
    const p0z = a.linvel().z + b.linvel().z;

    for (let i = 0; i < 120; i++) world.step();

    const p1x = a.linvel().x + b.linvel().x;
    const p1y = a.linvel().y + b.linvel().y;
    const p1z = a.linvel().z + b.linvel().z;

    expect(Math.abs(p1x - p0x)).toBeLessThan(3.0);
    expect(Math.abs(p1y - p0y)).toBeLessThan(1.0);
    expect(Math.abs(p1z - p0z)).toBeLessThan(1.0);
  });

  it('torque-free rotation preserves angular velocity (zero gravity)', () => {
    const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 0, 0)
    );
    world.createCollider(ColliderDesc3D.ball(1), body);
    body.setAngvel({ x: 0, y: 5, z: 0 });

    for (let i = 0; i < 60; i++) world.step();

    // Angular velocity magnitude should be preserved (~5 rad/s)
    const w = body.angvel();
    const wMag = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    expect(wMag).toBeCloseTo(5, 0); // within 0.5
  });

  it('constant velocity with no forces (zero gravity)', () => {
    const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 0, 0).setLinvel(2, 3, 4)
    );
    world.createCollider(ColliderDesc3D.ball(0.3), body);

    for (let i = 0; i < 60; i++) world.step();

    // After 1 second
    expect(body.translation().x).toBeCloseTo(2, 0);
    expect(body.translation().y).toBeCloseTo(3, 0);
    expect(body.translation().z).toBeCloseTo(4, 0);
  });
});
