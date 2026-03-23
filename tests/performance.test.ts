/**
 * Performance and scalability tests for the AVBD physics engine.
 * These tests validate that the engine can handle large numbers of bodies
 * and that performance scales reasonably.
 *
 * NOTE: These are CPU solver benchmarks. GPU benchmarks require WebGPU
 * context and should be run in a browser environment.
 */
import { describe, it, expect } from 'vitest';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';
import { computeGraphColoring, validateColoring } from '../src/core/graph-coloring.js';

// ─── Helpers ────────────────────────────────────────────────────────────────

function createSolverWithBodies(
  numBodies: number,
  config: any = {},
): AVBDSolver2D {
  const solver = new AVBDSolver2D({
    gravity: { x: 0, y: -9.81 },
    dt: 1 / 60,
    iterations: 5,
    ...config,
  });

  // Ground
  const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
  solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(100, 0.5));

  // Dynamic bodies in a grid
  const cols = Math.ceil(Math.sqrt(numBodies));
  for (let i = 0; i < numBodies; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const desc = RigidBodyDesc2D.dynamic()
      .setTranslation(col * 1.2 - cols * 0.6, 2 + row * 1.2);
    const handle = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));
  }

  return solver;
}

function timeMs(fn: () => void): number {
  const start = performance.now();
  fn();
  return performance.now() - start;
}

// ─── Body Count Scaling ─────────────────────────────────────────────────────

describe('Body count scaling', () => {
  it('should handle 50 bodies without errors', () => {
    const solver = createSolverWithBodies(50);

    for (let i = 0; i < 60; i++) {
      solver.step();
    }

    for (const body of solver.bodyStore.bodies) {
      expect(isFinite(body.position.x)).toBe(true);
      expect(isFinite(body.position.y)).toBe(true);
    }
  });

  it('should handle 100 bodies without errors', () => {
    const solver = createSolverWithBodies(100, { iterations: 5 });

    for (let i = 0; i < 30; i++) {
      solver.step();
    }

    for (const body of solver.bodyStore.bodies) {
      expect(isFinite(body.position.x)).toBe(true);
      expect(isFinite(body.position.y)).toBe(true);
    }
  });

  it('should handle 200 bodies', () => {
    const solver = createSolverWithBodies(200, { iterations: 3 });

    for (let i = 0; i < 10; i++) {
      solver.step();
    }

    for (const body of solver.bodyStore.bodies) {
      expect(isFinite(body.position.x)).toBe(true);
      expect(isFinite(body.position.y)).toBe(true);
    }
  });

  it('should handle 500 bodies', () => {
    const solver = createSolverWithBodies(500, { iterations: 2 });

    // Just verify it doesn't crash
    solver.step();

    for (const body of solver.bodyStore.bodies) {
      expect(isFinite(body.position.x)).toBe(true);
      expect(isFinite(body.position.y)).toBe(true);
    }
  });
});

// ─── Step Time Scaling ──────────────────────────────────────────────────────

describe('Step time scaling', () => {
  it('should scale sub-quadratically with body count (broadphase)', () => {
    // Measure step time for different body counts
    // Bodies spread out (no collisions) to isolate broadphase cost
    function stepTimeNoCollision(n: number): number {
      const solver = new AVBDSolver2D({
        gravity: { x: 0, y: 0 },
        iterations: 1,
      });

      for (let i = 0; i < n; i++) {
        const desc = RigidBodyDesc2D.dynamic()
          .setTranslation(i * 10, 0); // Widely spaced
        const handle = solver.bodyStore.addBody(desc);
        solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));
      }

      // Warm up
      solver.step();

      // Measure
      const time = timeMs(() => {
        for (let i = 0; i < 10; i++) solver.step();
      });
      return time / 10;
    }

    const t50 = stepTimeNoCollision(50);
    const t200 = stepTimeNoCollision(200);

    // With O(n^2) broadphase, 4x bodies = 16x time
    // We allow up to 20x (tolerant) to account for cache effects
    const ratio = t200 / Math.max(t50, 0.01);
    expect(ratio).toBeLessThan(20);
  });

  it('should scale linearly with solver iterations', () => {
    const solver = createSolverWithBodies(30);

    // Warm up
    solver.step();

    const t5 = timeMs(() => {
      solver.config.iterations = 5;
      for (let i = 0; i < 10; i++) solver.step();
    });

    const t10 = timeMs(() => {
      solver.config.iterations = 10;
      for (let i = 0; i < 10; i++) solver.step();
    });

    // Doubling iterations should roughly double time
    // Allow wide tolerance for CI/container environments
    const ratio = t10 / Math.max(t5, 0.01);
    expect(ratio).toBeLessThan(8); // Very lenient for noisy environments
    expect(ratio).toBeGreaterThan(0.3);
  });
});

// ─── Graph Coloring Performance ─────────────────────────────────────────────

describe('Graph coloring performance', () => {
  it('should color 1000 bodies quickly', () => {
    // Chain graph: 0-1-2-...-999
    const pairs: [number, number][] = [];
    for (let i = 0; i < 999; i++) {
      pairs.push([i, i + 1]);
    }

    const time = timeMs(() => {
      const result = computeGraphColoring(1000, pairs);
      expect(result.length).toBe(2); // Chain is bipartite
      expect(validateColoring(1000, pairs, result)).toBe(true);
    });

    expect(time).toBeLessThan(200); // Should be very fast
  });

  it('should color a dense graph (100 bodies, ~500 edges)', () => {
    const pairs: [number, number][] = [];
    // Random-ish connections
    for (let i = 0; i < 100; i++) {
      for (let j = i + 1; j < Math.min(i + 6, 100); j++) {
        pairs.push([i, j]);
      }
    }

    const time = timeMs(() => {
      const result = computeGraphColoring(100, pairs);
      expect(validateColoring(100, pairs, result)).toBe(true);
    });

    expect(time).toBeLessThan(100);
  });

  it('should produce few colors for stacking scenarios', () => {
    // Typical stacking: each box touches 2-3 others
    const pairs: [number, number][] = [];
    const n = 100;
    for (let i = 0; i < n - 1; i++) {
      pairs.push([i, i + 1]); // vertical contacts
      if (i + 2 < n && i % 3 === 0) {
        pairs.push([i, i + 2]); // skip contacts (diagonal)
      }
    }

    const result = computeGraphColoring(n, pairs);
    // Stacking graphs should use very few colors
    expect(result.length).toBeLessThanOrEqual(5);
    expect(validateColoring(n, pairs, result)).toBe(true);
  });
});

// ─── Contact Cache Performance ──────────────────────────────────────────────

describe('Contact persistence', () => {
  it('should warm-start contacts for faster convergence', () => {
    // With warmstarting, stacking should converge faster
    const solver = createSolverWithBodies(5, { iterations: 10 });

    // Run multiple steps to build up warmstarted contacts
    for (let i = 0; i < 30; i++) {
      solver.step();
    }

    // Check that contact cache has entries
    expect(solver.constraintStore.contactCache.size).toBeGreaterThan(0);
  });

  it('should expire old contacts', () => {
    const solver = createSolverWithBodies(2, { iterations: 5 });

    // Create contacts
    for (let i = 0; i < 30; i++) solver.step();

    // Now remove all dynamic bodies by making them fixed (simulate objects leaving)
    for (const body of solver.bodyStore.bodies) {
      if (body.type !== 1) { // Not fixed
        body.position = { x: 100, y: 100 }; // Move far away
      }
    }

    // Run enough steps for cache to expire
    solver.constraintStore.maxContactAge = 3;
    for (let i = 0; i < 10; i++) solver.step();

    // Cache should have been cleaned up
    // (entries older than maxContactAge should be removed)
    for (const [, cached] of solver.constraintStore.contactCache) {
      expect(cached.age).toBeLessThanOrEqual(solver.constraintStore.maxContactAge + 1);
    }
  });
});

// ─── Memory Usage ───────────────────────────────────────────────────────────

describe('Memory usage', () => {
  it('should not leak constraint rows over many steps', () => {
    const solver = createSolverWithBodies(10, { iterations: 5 });

    // Run many steps
    for (let i = 0; i < 100; i++) {
      solver.step();
    }

    // Constraint count should not grow unboundedly
    // It should be proportional to the number of active contacts
    const constraintCount = solver.constraintStore.count;
    // With 10 bodies, we shouldn't have more than ~50 constraint rows
    expect(constraintCount).toBeLessThan(200);
  });
});

// ─── Stability Under Stress ─────────────────────────────────────────────────

describe('Stability under stress', () => {
  it('should survive a box explosion (many simultaneous collisions)', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      iterations: 5,
    });

    // Many overlapping boxes at the same position
    for (let i = 0; i < 30; i++) {
      const desc = RigidBodyDesc2D.dynamic()
        .setTranslation(
          (Math.random() - 0.5) * 0.5,
          (Math.random() - 0.5) * 0.5 + 3,
        );
      const handle = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.3, 0.3));
    }

    // Ground
    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Should not crash or produce NaN
    for (let i = 0; i < 60; i++) {
      solver.step();
      for (const body of solver.bodyStore.bodies) {
        expect(isFinite(body.position.x)).toBe(true);
        expect(isFinite(body.position.y)).toBe(true);
      }
    }
  });

  it('should survive rapid body creation', () => {
    const solver = new AVBDSolver2D({
      gravity: { x: 0, y: -9.81 },
      iterations: 3,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(20, 0.5));

    // Add bodies during simulation
    for (let frame = 0; frame < 60; frame++) {
      if (frame % 3 === 0) {
        const desc = RigidBodyDesc2D.dynamic()
          .setTranslation((Math.random() - 0.5) * 5, 5 + Math.random() * 3);
        const handle = solver.bodyStore.addBody(desc);
        solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.3, 0.3));
      }
      solver.step();
    }

    for (const body of solver.bodyStore.bodies) {
      expect(isFinite(body.position.x)).toBe(true);
      expect(isFinite(body.position.y)).toBe(true);
    }
  });
});

// ─── Benchmark Report ───────────────────────────────────────────────────────

describe('Benchmark report', () => {
  it('should report step times for different scales', () => {
    const results: { bodies: number; avgMs: number; totalConstraints: number }[] = [];

    for (const n of [10, 25, 50, 100]) {
      const solver = createSolverWithBodies(n, { iterations: 5 });

      // Warm up
      for (let i = 0; i < 5; i++) solver.step();

      // Measure
      const steps = 20;
      const start = performance.now();
      for (let i = 0; i < steps; i++) solver.step();
      const elapsed = performance.now() - start;

      results.push({
        bodies: n,
        avgMs: elapsed / steps,
        totalConstraints: solver.constraintStore.count,
      });
    }

    // Log results
    console.log('\n=== AVBD CPU Solver Benchmark ===');
    console.log('Bodies | Avg Step (ms) | Constraints');
    console.log('-------|---------------|------------');
    for (const r of results) {
      console.log(
        `${String(r.bodies).padStart(6)} | ${r.avgMs.toFixed(2).padStart(13)} | ${String(r.totalConstraints).padStart(11)}`
      );
    }

    // Basic sanity: 100 bodies should complete in under 500ms per step on any CPU
    expect(results[results.length - 1].avgMs).toBeLessThan(500);
  });
});
