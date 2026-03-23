/**
 * 3D Solver correctness tests.
 *
 * Validates the AVBDSolver3D CPU solver produces correct physics
 * using the World3D API for proper body/collider setup.
 * These tests exercise the same algorithm that runs on the GPU.
 */
import { describe, it, expect } from 'vitest';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

function createWorld(overrides: Record<string, any> = {}) {
  return new World3D({ x: 0, y: -9.81, z: 0 }, {
    iterations: 10,
    useCPU: true,
    ...overrides,
  });
}

// ─── Free Fall ───────────────────────────────────────────────────────────────

describe('3D Solver: Free fall', () => {
  it('should fall under gravity (analytical match)', () => {
    const world = createWorld();
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    for (let i = 0; i < 60; i++) world.step();

    // y = y0 - 0.5*g*t^2 = 10 - 0.5*9.81*1 = 5.095
    const expected = 10 - 0.5 * 9.81 * 1;
    expect(body.translation().y).toBeCloseTo(expected, 0);
    expect(Math.abs(body.translation().x)).toBeLessThan(0.01);
    expect(Math.abs(body.translation().z)).toBeLessThan(0.01);
  });

  it('should not move fixed bodies', () => {
    const world = createWorld();
    const body = world.createRigidBody(
      RigidBodyDesc3D.fixed().setTranslation(1, 2, 3)
    );
    world.createCollider(ColliderDesc3D.cuboid(5, 0.5, 5), body);

    for (let i = 0; i < 30; i++) world.step();

    expect(body.translation()).toEqual({ x: 1, y: 2, z: 3 });
  });
});

// ─── Ground Collision ────────────────────────────────────────────────────────

describe('3D Solver: Box on ground', () => {
  it('should settle on ground surface', () => {
    const world = createWorld();
    // Ground: center y=0, half-height=0.5 => top at y=0.5
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 3, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    for (let i = 0; i < 120; i++) world.step();

    const y = body.translation().y;
    // Box should settle at y ≈ 1.0 (ground top 0.5 + box half-height 0.5)
    expect(y).toBeGreaterThan(0.3);
    expect(y).toBeLessThan(3.0);
  });

  it('sphere should settle on ground', () => {
    const world = createWorld();
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 3, 0)
    );
    world.createCollider(ColliderDesc3D.ball(0.5), body);

    for (let i = 0; i < 120; i++) world.step();

    const y = body.translation().y;
    expect(y).toBeGreaterThan(0.3);
    expect(y).toBeLessThan(3.0);
  });
});

// ─── Stacking ────────────────────────────────────────────────────────────────

describe('3D Solver: Stack stability', () => {
  it('should stack 3 boxes vertically', () => {
    const world = createWorld({ iterations: 15 });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

    const b1 = world.createRigidBody(RigidBodyDesc3D.dynamic().setTranslation(0, 1.5, 0));
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), b1);

    const b2 = world.createRigidBody(RigidBodyDesc3D.dynamic().setTranslation(0, 2.6, 0));
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), b2);

    const b3 = world.createRigidBody(RigidBodyDesc3D.dynamic().setTranslation(0, 3.7, 0));
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), b3);

    for (let i = 0; i < 180; i++) world.step();

    // Each box should be above the previous one
    expect(b1.translation().y).toBeGreaterThan(0.5);
    expect(b2.translation().y).toBeGreaterThan(b1.translation().y);
    expect(b3.translation().y).toBeGreaterThan(b2.translation().y);
  });
});

// ─── Sphere Collision ────────────────────────────────────────────────────────

describe('3D Solver: Sphere collision', () => {
  it('two spheres should not penetrate', () => {
    const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true });
    const r = 0.5;

    const s1 = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(-1, 0, 0).setLinvel(5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.ball(r), s1);

    const s2 = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(1, 0, 0).setLinvel(-5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.ball(r), s2);

    for (let i = 0; i < 60; i++) world.step();

    const p1 = s1.translation();
    const p2 = s2.translation();
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dz = p2.z - p1.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Should not penetrate
    expect(dist).toBeGreaterThanOrEqual(2 * r - 0.1);
  });
});

// ─── Friction Coupling ───────────────────────────────────────────────────────

describe('3D Solver: Friction coupling', () => {
  it('friction should slow down sliding body', () => {
    const world = createWorld();
    world.createCollider(
      ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.8)
    );

    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic()
        .setTranslation(0, 1.5, 0)
        .setLinvel(5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.8), body);

    // Step until settled
    for (let i = 0; i < 120; i++) world.step();

    // Body should have slowed down significantly due to friction
    const vx = body.linvel().x;
    expect(Math.abs(vx)).toBeLessThan(5); // Should have decelerated
  });
});

// ─── Velocity Recovery ───────────────────────────────────────────────────────

describe('3D Solver: Velocity recovery', () => {
  it('should recover velocity from position difference', () => {
    const world = createWorld();
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    world.step();

    // After one step, velocity should be approximately -g * dt
    expect(body.linvel().y).toBeLessThan(0);
    expect(Math.abs(body.linvel().y + 9.81 / 60)).toBeLessThan(0.05);
  });
});

// ─── Gravity Scale ───────────────────────────────────────────────────────────

describe('3D Solver: Gravity scale', () => {
  it('body with gravityScale=0 should not fall', () => {
    const world = createWorld();
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic()
        .setTranslation(0, 10, 0)
        .setGravityScale(0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    for (let i = 0; i < 60; i++) world.step();

    // Should stay at y=10 since gravity is disabled
    expect(body.translation().y).toBeCloseTo(10, 0);
  });
});
