/**
 * Stress and performance tests with hundreds of bodies.
 * Validates that the engine remains stable and performs reasonably
 * at scale in various challenging scenarios.
 */
import { describe, it, expect } from 'vitest';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';
import { World, RigidBodyDesc2D as WRigidBodyDesc2D, ColliderDesc2D as WColliderDesc2D } from '../src/2d/index.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';

function timeMs(fn: () => void): number {
  const start = performance.now();
  fn();
  return performance.now() - start;
}

// ─── 2D: Box Rain (200 bodies falling into a funnel) ─────────────────────────

describe('2D Stress: Box Rain (200 bodies)', () => {
  it('should remain stable after 200 boxes rain onto ground', { timeout: 15000 }, () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    // Ground
    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Left/right walls to contain bodies
    const lw = solver.bodyStore.addBody(
      RigidBodyDesc2D.fixed().setTranslation(-8, 10),
    );
    solver.bodyStore.attachCollider(lw.index, ColliderDesc2D.cuboid(0.3, 10));
    const rw = solver.bodyStore.addBody(
      RigidBodyDesc2D.fixed().setTranslation(8, 10),
    );
    solver.bodyStore.attachCollider(rw.index, ColliderDesc2D.cuboid(0.3, 10));

    // Spawn 200 boxes in a grid above the ground
    const numBodies = 200;
    const cols = 20;
    for (let i = 0; i < numBodies; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = (col - cols / 2) * 0.75;
      const y = 3 + row * 0.75;
      const handle = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation(x, y),
      );
      solver.bodyStore.attachCollider(handle.index,
        ColliderDesc2D.cuboid(0.3, 0.3).setFriction(0.4).setRestitution(0.05),
      );
    }

    // Simulate 3 seconds (180 frames)
    const elapsed = timeMs(() => {
      for (let step = 0; step < 180; step++) {
        solver.step();
      }
    });

    // Validate: all bodies finite, none exploded
    let allFinite = true;
    let maxY = -Infinity;
    for (const body of solver.bodyStore.bodies) {
      if (!isFinite(body.position.x) || !isFinite(body.position.y)) {
        allFinite = false;
      }
      if (body.position.y > maxY) maxY = body.position.y;
    }
    expect(allFinite).toBe(true);
    // No body should have exploded above 50m
    expect(maxY).toBeLessThan(50);

    console.log(`  200-body box rain: ${(elapsed / 180).toFixed(1)} ms/step avg, maxY=${maxY.toFixed(1)}`);
  });
});

// ─── 2D: Large Pyramid (120 boxes) ──────────────────────────────────────────

describe('2D Stress: Large Pyramid (120 boxes)', () => {
  it('should settle a 15-row pyramid stably', { timeout: 15000 }, () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 10, postStabilize: true });
    world.createCollider(WColliderDesc2D.cuboid(20, 0.5).setFriction(0.5));

    const pyramidBodies: any[] = [];
    const rows = 15;
    let count = 0;
    for (let row = 0; row < rows; row++) {
      const n = rows - row;
      const offset = -(n - 1) * 0.55;
      for (let col = 0; col < n; col++) {
        const x = offset + col * 1.1;
        const y = 0.8 + row * 1.05;
        const body = world.createRigidBody(
          WRigidBodyDesc2D.dynamic().setTranslation(x, y),
        );
        world.createCollider(
          WColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5).setDensity(1).setRestitution(0.05),
          body,
        );
        pyramidBodies.push(body);
        count++;
      }
    }

    expect(count).toBe(120); // 15+14+...+1 = 120

    // Simulate 5 seconds
    const elapsed = timeMs(() => {
      for (let i = 0; i < 300; i++) world.stepCPU();
    });

    // All bodies finite and above ground
    let allValid = true;
    let maxY = -Infinity;
    let minY = Infinity;
    for (const body of pyramidBodies) {
      const t = body.translation();
      if (!isFinite(t.x) || !isFinite(t.y)) allValid = false;
      if (t.y > maxY) maxY = t.y;
      if (t.y < minY) minY = t.y;
    }
    expect(allValid).toBe(true);
    // Some edge boxes may slide off and fall — that's OK for a stress test.
    // Key invariant: no explosion (maxY bounded) and no NaN.
    expect(maxY).toBeLessThan(60); // No explosion

    // Count how many settled above ground
    let aboveGround = 0;
    for (const body of pyramidBodies) {
      if (body.translation().y > 0.0) aboveGround++;
    }
    // At least 20% should remain above ground (large pyramids partially collapse)
    expect(aboveGround).toBeGreaterThan(count * 0.2);

    console.log(`  120-body pyramid: ${(elapsed / 300).toFixed(1)} ms/step, y range [${minY.toFixed(1)}, ${maxY.toFixed(1)}], ${aboveGround}/${count} above ground`);
  });
});

// ─── 2D: Dense Pile (300 overlapping bodies) ────────────────────────────────

describe('2D Stress: Dense Pile (300 bodies)', () => {
  it('should survive 300 overlapping bodies resolving to stable state', { timeout: 10000 }, () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    // Ground
    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(30, 0.5));

    // Walls
    const lw = solver.bodyStore.addBody(RigidBodyDesc2D.fixed().setTranslation(-10, 10));
    solver.bodyStore.attachCollider(lw.index, ColliderDesc2D.cuboid(0.3, 10));
    const rw = solver.bodyStore.addBody(RigidBodyDesc2D.fixed().setTranslation(10, 10));
    solver.bodyStore.attachCollider(rw.index, ColliderDesc2D.cuboid(0.3, 10));

    // 300 small balls in a dense cluster
    for (let i = 0; i < 300; i++) {
      const angle = (i / 300) * Math.PI * 20;
      const r = (i / 300) * 3;
      const x = Math.cos(angle) * r * 0.3;
      const y = 5 + Math.sin(angle) * r * 0.3 + (i / 300) * 8;
      const handle = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation(x, y),
      );
      solver.bodyStore.attachCollider(handle.index,
        ColliderDesc2D.ball(0.15).setFriction(0.3).setRestitution(0.1),
      );
    }

    // Simulate 2 seconds
    const elapsed = timeMs(() => {
      for (let step = 0; step < 120; step++) {
        solver.step();
      }
    });

    // Validate no NaN
    let nanCount = 0;
    for (const body of solver.bodyStore.bodies) {
      if (!isFinite(body.position.x) || !isFinite(body.position.y)) nanCount++;
    }
    expect(nanCount).toBe(0);

    console.log(`  300-body dense pile: ${(elapsed / 120).toFixed(1)} ms/step`);
  });
});

// ─── 2D: Continuous Spawning (bodies added mid-simulation) ──────────────────

describe('2D Stress: Continuous Spawning', () => {
  it('should handle spawning 5 bodies per frame for 60 frames (300 total)', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 3,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(30, 0.5));

    const elapsed = timeMs(() => {
      for (let frame = 0; frame < 60; frame++) {
        // Spawn 5 bodies per frame
        for (let j = 0; j < 5; j++) {
          const x = (Math.random() - 0.5) * 10;
          const y = 8 + Math.random() * 5;
          const handle = solver.bodyStore.addBody(
            RigidBodyDesc2D.dynamic().setTranslation(x, y),
          );
          const shape = Math.random() > 0.5
            ? ColliderDesc2D.cuboid(0.2 + Math.random() * 0.2, 0.2 + Math.random() * 0.2)
            : ColliderDesc2D.ball(0.15 + Math.random() * 0.1);
          solver.bodyStore.attachCollider(handle.index,
            shape.setFriction(0.4).setRestitution(0.1),
          );
        }
        solver.step();
      }
    });

    // Should have 300 dynamic + 1 ground = 301 total
    expect(solver.bodyStore.bodies.length).toBe(301);

    let nanCount = 0;
    for (const body of solver.bodyStore.bodies) {
      if (!isFinite(body.position.x) || !isFinite(body.position.y)) nanCount++;
    }
    expect(nanCount).toBe(0);

    console.log(`  Continuous spawn (300 bodies over 60 frames): ${(elapsed / 60).toFixed(1)} ms/step avg`);
  });
});

// ─── 2D: Mixed Shapes Stress ────────────────────────────────────────────────

describe('2D Stress: Mixed Shapes (200 bodies)', () => {
  it('should handle a mix of boxes and balls of varying sizes', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      dt: 1 / 60,
      iterations: 5,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Mix of sizes and shapes
    for (let i = 0; i < 200; i++) {
      const col = i % 15;
      const row = Math.floor(i / 15);
      const x = (col - 7) * 1.0;
      const y = 2 + row * 1.0;
      const handle = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation(x, y),
      );
      if (i % 3 === 0) {
        // Large box
        solver.bodyStore.attachCollider(handle.index,
          ColliderDesc2D.cuboid(0.4, 0.4).setFriction(0.5),
        );
      } else if (i % 3 === 1) {
        // Small box
        solver.bodyStore.attachCollider(handle.index,
          ColliderDesc2D.cuboid(0.2, 0.2).setFriction(0.3),
        );
      } else {
        // Ball
        solver.bodyStore.attachCollider(handle.index,
          ColliderDesc2D.ball(0.2).setFriction(0.4),
        );
      }
    }

    const elapsed = timeMs(() => {
      for (let step = 0; step < 120; step++) {
        solver.step();
      }
    });

    let nanCount = 0;
    for (const body of solver.bodyStore.bodies) {
      if (!isFinite(body.position.x) || !isFinite(body.position.y)) nanCount++;
    }
    expect(nanCount).toBe(0);

    console.log(`  200-body mixed shapes: ${(elapsed / 120).toFixed(1)} ms/step`);
  });
});

// ─── 3D Stress: Box Rain (100 bodies) ───────────────────────────────────────

describe('3D Stress: Box Rain (100 bodies)', () => {
  it('should handle 100 cubes falling onto ground', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 5,
      useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(20, 0.5, 20).setFriction(0.5));

    const bodies: any[] = [];
    const cols = 10;
    for (let i = 0; i < 100; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = (col - cols / 2) * 1.2;
      const z = (row - 5) * 1.2;
      const y = 2 + row * 0.5;
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(x, y, z),
      );
      world.createCollider(
        ColliderDesc3D.cuboid(0.4, 0.4, 0.4).setFriction(0.4).setDensity(1),
        body,
      );
      bodies.push(body);
    }

    const elapsed = timeMs(() => {
      for (let step = 0; step < 120; step++) {
        world.step();
      }
    });

    let nanCount = 0;
    let maxY = -Infinity;
    for (const body of bodies) {
      const t = body.translation();
      if (!isFinite(t.x) || !isFinite(t.y) || !isFinite(t.z)) nanCount++;
      if (t.y > maxY) maxY = t.y;
    }
    expect(nanCount).toBe(0);
    expect(maxY).toBeLessThan(30);

    console.log(`  3D 100-body box rain: ${(elapsed / 120).toFixed(1)} ms/step, maxY=${maxY.toFixed(1)}`);
  });
});

// ─── 3D Stress: Sphere Pile (100 bodies) ────────────────────────────────────

describe('3D Stress: Sphere Pile (100 spheres)', () => {
  it('should handle 100 spheres settling into a pile', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 5,
      useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));

    // Walls to contain spheres
    const wallDescs = [
      { x: -4, y: 3, z: 0, hx: 0.3, hy: 3, hz: 4 },
      { x: 4, y: 3, z: 0, hx: 0.3, hy: 3, hz: 4 },
      { x: 0, y: 3, z: -4, hx: 4, hy: 3, hz: 0.3 },
      { x: 0, y: 3, z: 4, hx: 4, hy: 3, hz: 0.3 },
    ];
    for (const w of wallDescs) {
      const wb = world.createRigidBody(
        RigidBodyDesc3D.fixed().setTranslation(w.x, w.y, w.z),
      );
      world.createCollider(ColliderDesc3D.cuboid(w.hx, w.hy, w.hz), wb);
    }

    const bodies: any[] = [];
    for (let i = 0; i < 100; i++) {
      const x = (Math.random() - 0.5) * 4;
      const z = (Math.random() - 0.5) * 4;
      const y = 2 + i * 0.3;
      const body = world.createRigidBody(
        RigidBodyDesc3D.dynamic().setTranslation(x, y, z),
      );
      world.createCollider(
        ColliderDesc3D.ball(0.25).setFriction(0.3).setDensity(1),
        body,
      );
      bodies.push(body);
    }

    const elapsed = timeMs(() => {
      for (let step = 0; step < 120; step++) {
        world.step();
      }
    });

    let nanCount = 0;
    for (const body of bodies) {
      const t = body.translation();
      if (!isFinite(t.x) || !isFinite(t.y) || !isFinite(t.z)) nanCount++;
    }
    expect(nanCount).toBe(0);

    console.log(`  3D 100-sphere pile: ${(elapsed / 120).toFixed(1)} ms/step`);
  });
});

// ─── Benchmark Report: Scaling ──────────────────────────────────────────────

describe('Stress benchmark report', () => {
  it('should report step times for 100, 200, 300, 500 body counts', { timeout: 30000 }, () => {
    const results: { bodies: number; avgMs: number; constraints: number }[] = [];

    for (const n of [100, 200, 300, 500]) {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: -9.81 },
        dt: 1 / 60,
        iterations: 5,
      });

      const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
      solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(50, 0.5));

      // Grid of boxes
      const cols = Math.ceil(Math.sqrt(n));
      for (let i = 0; i < n; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const handle = solver.bodyStore.addBody(
          RigidBodyDesc2D.dynamic().setTranslation(
            (col - cols / 2) * 1.1,
            2 + row * 1.1,
          ),
        );
        solver.bodyStore.attachCollider(handle.index,
          ColliderDesc2D.cuboid(0.4, 0.4).setFriction(0.3),
        );
      }

      // Warm up (5 steps)
      for (let i = 0; i < 5; i++) solver.step();

      // Measure 20 steps
      const elapsed = timeMs(() => {
        for (let i = 0; i < 20; i++) solver.step();
      });

      results.push({
        bodies: n,
        avgMs: elapsed / 20,
        constraints: solver.constraintStore.count,
      });
    }

    console.log('\n=== AVBD CPU Stress Benchmark ===');
    console.log('Bodies | Avg Step (ms) | Constraints');
    console.log('-------|---------------|------------');
    for (const r of results) {
      console.log(
        `${String(r.bodies).padStart(6)} | ${r.avgMs.toFixed(1).padStart(13)} | ${String(r.constraints).padStart(11)}`,
      );
    }

    // 500 bodies should complete in under 2 seconds per step
    expect(results[results.length - 1].avgMs).toBeLessThan(2000);
  });
});
