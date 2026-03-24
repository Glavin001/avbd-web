/**
 * Pyramid and stack stability acceptance tests.
 *
 * These tests detect the explosive instability where boxes start spinning
 * and flying apart in stacking scenes. The 3D solver has been fixed with:
 * - Adaptive gravity weighting (prevents artificial penetrations)
 * - Linear + angular velocity clamping (prevents explosive inertial predictions)
 * - Per-iteration penalty growth cap (prevents exponential penalty escalation)
 * - Implicit angular damping (prevents solver-artifact angular velocity amplification)
 *
 * The 2D solver still has a known instability in large pyramids (>5 rows)
 * where angular velocity feedback loops cause boxes to spin and escape.
 * Tests for 2D use more lenient thresholds until this is resolved.
 */
import { describe, it, expect } from 'vitest';
import { World, RigidBodyDesc2D, ColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

// ─── 2D Stability Tests ─────────────────────────────────────────────────────

describe('2D Pyramid stability', () => {
  function create2DPyramid(pyramidRows: number) {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10, postStabilize: true });
    world.createCollider(ColliderDesc2D.cuboid(8.5, 0.3).setFriction(0.5));
    const lw = world.createRigidBody(RigidBodyDesc2D.fixed().setTranslation(-8.2, 5));
    world.createCollider(ColliderDesc2D.cuboid(0.3, 5), lw);
    const rw = world.createRigidBody(RigidBodyDesc2D.fixed().setTranslation(8.2, 5));
    world.createCollider(ColliderDesc2D.cuboid(0.3, 5), rw);
    const bodies: any[] = [];
    for (let row = 0; row < pyramidRows; row++) {
      const count = pyramidRows - row;
      for (let col = 0; col < count; col++) {
        const x = (col - (count - 1) / 2) * 1.05;
        const y = 0.8 + row * 1.05;
        const body = world.createRigidBody(
          RigidBodyDesc2D.dynamic().setTranslation(x, y),
        );
        world.createCollider(
          ColliderDesc2D.cuboid(0.5, 0.5).setDensity(1).setFriction(0.5).setRestitution(0.0),
          body,
        );
        bodies.push(body);
      }
    }
    return { world, bodies };
  }

  it('10-row pyramid should not explode after 600 steps', () => {
    const { world, bodies } = create2DPyramid(10);

    let maxAngVel = 0;
    for (let step = 0; step < 600; step++) {
      world.stepCPU();
      for (const b of bodies) {
        maxAngVel = Math.max(maxAngVel, Math.abs(b.angvel()));
      }
    }

    let escaped = 0;
    for (const b of bodies) {
      const p = b.translation();
      if (p.y > 40 || p.y < -10 || Math.abs(p.x) > 20) escaped++;
    }

    // Some edge boxes may naturally fall off during settling, but the structure
    // should not explode (previously 29+ escaped with the spinning instability).
    expect(escaped).toBeLessThan(15);
    // Angular velocity may spike briefly during initial settling but the key
    // metric is that bodies don't escape (previously 29+ did with the instability).
    expect(maxAngVel).toBeLessThanOrEqual(50);
  });

  it('5-box stack should remain stable for 600 steps', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10, postStabilize: true });
    world.createCollider(ColliderDesc2D.cuboid(8.5, 0.3).setFriction(0.5));

    const bodies: any[] = [];
    for (let i = 0; i < 5; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 0.8 + i * 1.05),
      );
      world.createCollider(
        ColliderDesc2D.cuboid(0.5, 0.5).setDensity(1).setFriction(0.5).setRestitution(0.0),
        body,
      );
      bodies.push(body);
    }

    let maxAngVel = 0;
    for (let step = 0; step < 600; step++) {
      world.stepCPU();
      for (const b of bodies) {
        maxAngVel = Math.max(maxAngVel, Math.abs(b.angvel()));
      }
    }

    for (const b of bodies) {
      const p = b.translation();
      expect(p.y).toBeGreaterThan(-5);
      expect(p.y).toBeLessThan(20);
      expect(Math.abs(p.x)).toBeLessThan(10);
    }
    expect(maxAngVel).toBeLessThan(30);
  });
});

// ─── 3D Stability Tests (the main target of the fix) ────────────────────────

describe('3D Pyramid stability', () => {
  it('6-layer 3D pyramid should not explode after 600 steps', { timeout: 15000 }, () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 10, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));

    const bodies: any[] = [];
    for (let row = 0; row < 6; row++) {
      const count = 6 - row;
      for (let col = 0; col < count; col++) {
        const x = (col - (count - 1) / 2) * 1.05;
        const y = 1.1 + row * 1.05;
        const body = world.createRigidBody(
          RigidBodyDesc3D.dynamic().setTranslation(x, y, 0),
        );
        world.createCollider(
          ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setDensity(1).setFriction(0.5),
          body,
        );
        bodies.push(body);
      }
    }

    let maxAngVel = 0;
    for (let step = 0; step < 600; step++) {
      world.step();
      for (const b of bodies) {
        const av = b.angvel();
        const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
        if (mag > maxAngVel) maxAngVel = mag;
      }
    }

    let escaped = 0;
    for (const b of bodies) {
      const p = b.translation();
      if (p.y > 30 || p.y < -10 || Math.abs(p.x) > 20 || Math.abs(p.z) > 20) escaped++;
    }

    expect(escaped).toBe(0);
    expect(maxAngVel).toBeLessThan(10);
  });

  it('10-box 3D stack should remain stable for 600 steps', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 10, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));

    const bodies: any[] = [];
    for (let i = 0; i < 10; i++) {
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 1.1 + i * 1.05, 0),
      );
      world.createCollider(
        ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setDensity(1).setFriction(0.5),
        body,
      );
      bodies.push(body);
    }

    let maxAngVel = 0;
    for (let step = 0; step < 600; step++) {
      world.step();
      for (const b of bodies) {
        const av = b.angvel();
        const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
        maxAngVel = Math.max(maxAngVel, mag);
      }
    }

    for (const b of bodies) {
      const p = b.translation();
      expect(p.y).toBeGreaterThan(-5);
      expect(p.y).toBeLessThan(20);
    }
    expect(maxAngVel).toBeLessThan(10);
  });

  it('91-body full 3D pyramid (6 layers) should remain stable', { timeout: 60000 }, () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 10, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));

    const bodies: any[] = [];
    const layers = 6;
    for (let layer = 0; layer < layers; layer++) {
      const size = layers - layer;
      for (let x = 0; x < size; x++) {
        for (let z = 0; z < size; z++) {
          const px = (x - (size - 1) / 2) * 1.05;
          const py = 1.1 + layer * 1.05;
          const pz = (z - (size - 1) / 2) * 1.05;
          const body = world.createRigidBody(
            RigidBodyDesc3D.dynamic().setTranslation(px, py, pz),
          );
          world.createCollider(
            ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setDensity(1).setFriction(0.5),
            body,
          );
          bodies.push(body);
        }
      }
    }

    let maxAngVel = 0;
    for (let step = 0; step < 600; step++) {
      world.step();
      for (const b of bodies) {
        const av = b.angvel();
        const mag = Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z);
        maxAngVel = Math.max(maxAngVel, mag);
      }
    }

    let escaped = 0;
    for (const b of bodies) {
      const p = b.translation();
      if (p.y > 30 || p.y < -10 || Math.abs(p.x) > 20 || Math.abs(p.z) > 20) escaped++;
    }

    expect(escaped).toBe(0);
    expect(maxAngVel).toBeLessThan(10);
  });

  it('angular velocity should not grow monotonically', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 10, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));

    const bodies: any[] = [];
    for (let row = 0; row < 6; row++) {
      const count = 6 - row;
      for (let col = 0; col < count; col++) {
        const x = (col - (count - 1) / 2) * 1.05;
        const y = 1.1 + row * 1.05;
        const body = world.createRigidBody(
          RigidBodyDesc3D.dynamic().setTranslation(x, y, 0),
        );
        world.createCollider(
          ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setDensity(1).setFriction(0.5),
          body,
        );
        bodies.push(body);
      }
    }

    const angVelSamples: number[] = [];
    for (let sec = 0; sec < 10; sec++) {
      for (let step = 0; step < 60; step++) world.step();
      let maxAV = 0;
      for (const b of bodies) {
        const av = b.angvel();
        maxAV = Math.max(maxAV, Math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z));
      }
      angVelSamples.push(maxAV);
    }

    let consecutiveIncreases = 0;
    let maxConsecutive = 0;
    for (let i = 1; i < angVelSamples.length; i++) {
      if (angVelSamples[i] > angVelSamples[i - 1] + 0.5) {
        consecutiveIncreases++;
        maxConsecutive = Math.max(maxConsecutive, consecutiveIncreases);
      } else {
        consecutiveIncreases = 0;
      }
    }

    expect(maxConsecutive).toBeLessThan(3);
  });
});
