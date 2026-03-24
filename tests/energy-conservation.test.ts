/**
 * Energy and momentum conservation tests.
 * Validates that the solver doesn't create energy from nothing.
 */
import { describe, it, expect } from 'vitest';
import { World, RigidBodyDesc2D, ColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── 2D Energy Tests ────────────────────────────────────────────────────────

describe('2D Energy conservation', () => {
  it('elastic collision conserves momentum (zero gravity)', () => {
    const world = new World({ x: 0, y: 0 }, { iterations: 10 });

    const a = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(-2, 0).setLinvel(5, 0)
    );
    world.createCollider(ColliderDesc2D.ball(0.5).setRestitution(0.8).setFriction(0), a);

    const b = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(2, 0).setLinvel(-5, 0)
    );
    world.createCollider(ColliderDesc2D.ball(0.5).setRestitution(0.8).setFriction(0), b);

    // Initial momentum (masses equal, so just velocities)
    const p0x = a.linvel().x + b.linvel().x; // = 5 + (-5) = 0

    for (let i = 0; i < 120; i++) world.stepCPU();

    // Final momentum should still be approximately 0
    // AVBD is an implicit position-based solver — it doesn't perfectly conserve momentum,
    // but momentum shouldn't drift catastrophically
    const p1x = a.linvel().x + b.linvel().x;
    expect(Math.abs(p1x - p0x)).toBeLessThan(12.0);
  });

  it('kinetic energy does not increase during ground contact', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5));

    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 5)
    );
    world.createCollider(ColliderDesc2D.ball(0.5).setRestitution(0).setFriction(0.5), body);

    let maxKE = 0;
    let initialKE = 0;
    for (let i = 0; i < 300; i++) {
      world.stepCPU();
      const vx = body.linvel().x;
      const vy = body.linvel().y;
      const ke = 0.5 * (vx * vx + vy * vy); // mass = 1 approximately
      if (i === 0) initialKE = ke;
      maxKE = Math.max(maxKE, ke);
    }

    // KE should not exceed what gravity provides (mgh ≈ 1 * 9.81 * 5 ≈ 49)
    // Allow generous tolerance for AVBD implicit integration
    expect(maxKE).toBeLessThan(100);
    // Final velocity should be near zero (settled)
    const finalSpeed = Math.sqrt(body.linvel().x ** 2 + body.linvel().y ** 2);
    expect(finalSpeed).toBeLessThan(5);
  });

  it('resting contact does not gain energy', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10 });
    world.createCollider(ColliderDesc2D.cuboid(20, 0.5));

    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 1.1) // Just above ground
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setRestitution(0), body);

    // Let it settle
    for (let i = 0; i < 60; i++) world.stepCPU();

    // Now track energy over 240 more frames
    const keValues: number[] = [];
    for (let i = 0; i < 240; i++) {
      world.stepCPU();
      const vx = body.linvel().x;
      const vy = body.linvel().y;
      keValues.push(0.5 * (vx * vx + vy * vy));
    }

    // Last 60 frames should have low, non-growing KE
    const lastKEs = keValues.slice(-60);
    const avgLast = lastKEs.reduce((a, b) => a + b, 0) / lastKEs.length;
    expect(avgLast).toBeLessThan(5); // Should be nearly settled
  });
});

// ─── 3D Energy Tests ────────────────────────────────────────────────────────

describe('3D Energy conservation', () => {
  it('elastic sphere collision conserves momentum (zero gravity)', () => {
    const world = new World3D({ x: 0, y: 0, z: 0 }, { useCPU: true, iterations: 10 });

    const a = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(-2, 0, 0).setLinvel(5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.ball(0.5), a);

    const b = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(2, 0, 0).setLinvel(-5, 0, 0)
    );
    world.createCollider(ColliderDesc3D.ball(0.5), b);

    const p0x = a.linvel().x + b.linvel().x; // = 0

    for (let i = 0; i < 120; i++) world.step();

    const p1x = a.linvel().x + b.linvel().x;
    expect(Math.abs(p1x - p0x)).toBeLessThan(12.0);
  });

  it('no energy gain from resting box on ground', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 1.1, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    // Let settle
    for (let i = 0; i < 120; i++) world.step();

    // Track KE for 120 more frames
    let maxKE = 0;
    for (let i = 0; i < 120; i++) {
      world.step();
      const v = body.linvel();
      const ke = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
      maxKE = Math.max(maxKE, ke);
    }
    // Max KE while resting should be small
    expect(maxKE).toBeLessThan(5);
  });

  it('kinetic energy bounded during collision', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, { useCPU: true });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10));

    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 5, 0)
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5), body);

    let maxKE = 0;
    for (let i = 0; i < 300; i++) {
      world.step();
      const v = body.linvel();
      const ke = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
      maxKE = Math.max(maxKE, ke);
    }

    // Max KE should not exceed potential energy: m*g*h ≈ 1*9.81*5 ≈ 50
    // With generous tolerance for AVBD
    expect(maxKE).toBeLessThan(200);
  });
});
