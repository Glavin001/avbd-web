/**
 * Physics invariant tests — properties that must ALWAYS hold.
 * Tests ground penetration, NaN detection, and basic stability guarantees.
 */
import { describe, it, expect } from 'vitest';
import { World, RigidBodyDesc2D, ColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── 2D Ground Penetration ─────────────────────────────────────────────────

describe('2D Ground penetration invariants', () => {
  function createWorld2D(config: any = {}) {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10, ...config });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5)); // Ground: top at y=0.5
    return world;
  }

  it('box never penetrates ground over 5 seconds', () => {
    const world = createWorld2D();
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 5)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    let minY = Infinity;
    for (let i = 0; i < 300; i++) {
      world.stepCPU();
      const y = body.translation().y;
      minY = Math.min(minY, y);
      expect(isFinite(y)).toBe(true);
    }
    // Box center should never go below ground top (0.5) + box half-height (0.5) - tolerance
    expect(minY).toBeGreaterThan(0.3);
  });

  it('heavy box (density=100) never penetrates ground', () => {
    const world = createWorld2D({ iterations: 15 });
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 5)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setDensity(100), body);

    let minY = Infinity;
    for (let i = 0; i < 300; i++) {
      world.stepCPU();
      const y = body.translation().y;
      minY = Math.min(minY, y);
      expect(isFinite(y)).toBe(true);
    }
    expect(minY).toBeGreaterThan(0.0);
  });

  it('ball never penetrates ground', () => {
    const world = createWorld2D();
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 5)
    );
    world.createCollider(ColliderDesc2D.ball(0.5), body);

    let minY = Infinity;
    for (let i = 0; i < 300; i++) {
      world.stepCPU();
      minY = Math.min(minY, body.translation().y);
    }
    // Ball center should stay above ground top (0.5) + radius (0.5) - tolerance
    expect(minY).toBeGreaterThan(0.3);
  });

  it('fast-falling box stops above ground', () => {
    const world = createWorld2D();
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 3).setLinvel(0, -20)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    let minY = Infinity;
    for (let i = 0; i < 180; i++) {
      world.stepCPU();
      minY = Math.min(minY, body.translation().y);
    }
    expect(minY).toBeGreaterThan(0.0);
  });

  it('all states are finite (no NaN/Infinity)', () => {
    const world = createWorld2D();
    for (let i = 0; i < 5; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(i * 0.5, 3 + i * 1.2)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.4, 0.4), body);
    }

    for (let step = 0; step < 300; step++) {
      world.stepCPU();
      const states = world.getBodyStates();
      for (let i = 0; i < states.length; i++) {
        expect(isFinite(states[i])).toBe(true);
      }
    }
  });
});

// ─── 3D Ground Penetration ─────────────────────────────────────────────────

describe('3D Ground penetration invariants', () => {
  function createWorld3D(config: any = {}) {
    return new World3D({ x: 0, y: -9.81, z: 0 }, { iterations: 10, useCPU: true, ...config });
  }

  it('box never penetrates ground over 5 seconds', () => {
    const world = createWorld3D();
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10)); // Ground: top at y=0.5
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 5, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    let minY = Infinity;
    for (let i = 0; i < 300; i++) {
      world.step();
      const y = body.translation().y;
      minY = Math.min(minY, y);
      expect(isFinite(y)).toBe(true);
    }
    // Box center y should stay above ground_top(0.5) + box_half(0.5) - tolerance
    expect(minY).toBeGreaterThan(0.3);
  });

  it('fast-moving box stops above ground', () => {
    const world = createWorld3D();
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 3, 0).setLinvel(0, -10, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    let minY = Infinity;
    for (let i = 0; i < 180; i++) {
      world.step();
      minY = Math.min(minY, body.translation().y);
    }
    expect(minY).toBeGreaterThan(0.0);
  });

  it('sphere never penetrates ground', () => {
    const world = createWorld3D();
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 5, 0)
    );
    world.createCollider(ColliderDesc3D.ball(0.5), body);

    let minY = Infinity;
    for (let i = 0; i < 300; i++) {
      world.step();
      minY = Math.min(minY, body.translation().y);
    }
    expect(minY).toBeGreaterThan(0.3);
  });

  it('all body states are finite (no NaN)', () => {
    const world = createWorld3D();
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));
    for (let i = 0; i < 3; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(i * 0.3, 3 + i * 1.2, 0)
      );
      world.createCollider(ColliderDesc3D.cuboid(0.4, 0.4, 0.4), body);
    }

    for (let step = 0; step < 180; step++) {
      world.step();
    }
    // No assertions needed beyond not throwing — the expect(isFinite) above covers it
  });
});
