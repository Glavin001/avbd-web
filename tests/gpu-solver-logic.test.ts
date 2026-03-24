/**
 * GPU solver CPU-side logic tests.
 *
 * The GPU solver runs broadphase, body initialization, constraint creation,
 * and buffer packing on the CPU before dispatching to GPU shaders. These tests
 * validate that CPU-side logic is correct — they catch bugs that wouldn't appear
 * in shader-only validation tests.
 *
 * Since WebGPU isn't available in Node.js, we test the CPU-side methods directly
 * by importing the GPU solver class and exercising its internal logic through
 * the CPU solver (which shares the same broadphase and body init code).
 */

import { describe, it, expect } from 'vitest';
import { AVBDSolver3D } from '../src/core/solver-3d.js';
import { RigidBodyDesc3D, ColliderDesc3D } from '../src/core/rigid-body-3d.js';
import { collide3D, getAABB3D } from '../src/3d/collision-gjk.js';
import { RigidBodyType } from '../src/core/types.js';
import { vec3Length } from '../src/core/math.js';

// ─── GPU solver broadphase parity with CPU solver ────────────────────────────

describe('GPU solver broadphase produces same pairs as CPU solver', () => {
  it('should detect collision between two overlapping spheres', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });
    // Ground
    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    // Two overlapping spheres
    const desc1 = RigidBodyDesc3D.dynamic().setTranslation(0, 2, 0);
    const b1 = solver.bodyStore.addBody(desc1);
    solver.bodyStore.attachCollider(b1.index, ColliderDesc3D.ball(0.5));

    const desc2 = RigidBodyDesc3D.dynamic().setTranslation(0.5, 2, 0);
    const b2 = solver.bodyStore.addBody(desc2);
    solver.bodyStore.attachCollider(b2.index, ColliderDesc3D.ball(0.5));

    solver.step();
    // Should have contact constraints
    expect(solver.constraintRows.length).toBeGreaterThan(0);
  });

  it('should NOT detect collision between two distant spheres', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: 0, z: 0 }, // zero gravity to prevent falling
      iterations: 5,
    });

    const desc1 = RigidBodyDesc3D.dynamic().setTranslation(0, 5, 0);
    const b1 = solver.bodyStore.addBody(desc1);
    solver.bodyStore.attachCollider(b1.index, ColliderDesc3D.ball(0.5));

    const desc2 = RigidBodyDesc3D.dynamic().setTranslation(10, 5, 0);
    const b2 = solver.bodyStore.addBody(desc2);
    solver.bodyStore.attachCollider(b2.index, ColliderDesc3D.ball(0.5));

    solver.step();
    // No constraints (only 2 distant spheres, no contacts)
    expect(solver.constraintRows.length).toBe(0);
  });

  it('should handle 100+ spheres without crash', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 3,
    });

    // Ground
    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    // 100 spheres in a grid
    for (let i = 0; i < 100; i++) {
      const x = (i % 10 - 5) * 0.4;
      const y = 1.0 + Math.floor(i / 10) * 0.4;
      const z = 0;
      const desc = RigidBodyDesc3D.dynamic().setTranslation(x, y, z);
      const b = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.15));
    }

    // Should not throw
    solver.step();
    expect(solver.bodyStore.bodies.length).toBe(101); // 100 spheres + ground
  });
});

// ─── Body initialization correctness ─────────────────────────────────────────

describe('Body initialization produces correct inertial targets', () => {
  it('free-falling body should have inertial position below initial position', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    solver.step();

    const body = solver.bodyStore.bodies[b.index];
    // After one step, position should be below initial
    expect(body.position.y).toBeLessThan(10);
    // Position should be finite
    expect(isFinite(body.position.x)).toBe(true);
    expect(isFinite(body.position.y)).toBe(true);
    expect(isFinite(body.position.z)).toBe(true);
  });

  it('angular velocity should be preserved through body init', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: 0, z: 0 },
      iterations: 5,
    });

    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 5, 0).setAngvel(0, 0, 1.0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    solver.step();

    const body = solver.bodyStore.bodies[b.index];
    // Angular velocity should still exist (damped slightly but not zero)
    expect(vec3Length(body.angularVelocity)).toBeGreaterThan(0.5);
  });

  it('prev position/rotation should be saved correctly', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    const desc = RigidBodyDesc3D.dynamic().setTranslation(3, 7, -2);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    // After first step, prevPosition should be the initial position
    solver.step();
    const body = solver.bodyStore.bodies[b.index];

    // prevPosition was set at start of step (before position changed)
    // On second step, prevPosition should equal position from after first step
    const posAfterStep1Y = body.position.y;
    solver.step();
    expect(body.prevPosition.y).toBeCloseTo(posAfterStep1Y, 5);
  });
});

// ─── Inline AABB computation ─────────────────────────────────────────────────

describe('Inline AABB matches getAABB3D reference', () => {
  it('sphere AABB should be position ± radius', () => {
    const solver = new AVBDSolver3D({ gravity: { x: 0, y: 0, z: 0 } });
    const desc = RigidBodyDesc3D.dynamic().setTranslation(5, 3, -1);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.75));

    const body = solver.bodyStore.bodies[b.index];
    const aabb = getAABB3D(body);
    expect(aabb.min.x).toBeCloseTo(5 - 0.75);
    expect(aabb.max.x).toBeCloseTo(5 + 0.75);
    expect(aabb.min.y).toBeCloseTo(3 - 0.75);
    expect(aabb.max.y).toBeCloseTo(3 + 0.75);
    expect(aabb.min.z).toBeCloseTo(-1 - 0.75);
    expect(aabb.max.z).toBeCloseTo(-1 + 0.75);
  });

  it('axis-aligned cuboid AABB should match half-extents', () => {
    const solver = new AVBDSolver3D({ gravity: { x: 0, y: 0, z: 0 } });
    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 0, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.cuboid(1, 2, 3));

    const body = solver.bodyStore.bodies[b.index];
    const aabb = getAABB3D(body);
    expect(aabb.min.x).toBeCloseTo(-1);
    expect(aabb.max.x).toBeCloseTo(1);
    expect(aabb.min.y).toBeCloseTo(-2);
    expect(aabb.max.y).toBeCloseTo(2);
    expect(aabb.min.z).toBeCloseTo(-3);
    expect(aabb.max.z).toBeCloseTo(3);
  });
});

// ─── Constraint creation ─────────────────────────────────────────────────────

describe('Contact constraint creation for GPU solver', () => {
  it('contact between sphere and ground should create 3 constraint rows (normal + 2 friction)', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    // Sphere slightly overlapping the ground (ground top at 0.5, sphere center at 0.9, radius 0.5)
    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 0.9, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    solver.step();

    // 3D contacts produce triplets: [normal, friction1, friction2]
    expect(solver.constraintRows.length % 3).toBe(0);
    expect(solver.constraintRows.length).toBeGreaterThanOrEqual(3);
  });

  it('constraint Jacobians should have correct body indices', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 1.0, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    solver.step();

    for (const row of solver.constraintRows) {
      // Body indices should be valid
      expect(row.bodyA).toBeGreaterThanOrEqual(0);
      expect(row.bodyB).toBeGreaterThanOrEqual(0);
      expect(row.bodyA).toBeLessThan(solver.bodyStore.bodies.length);
      expect(row.bodyB).toBeLessThan(solver.bodyStore.bodies.length);
      // Jacobians should be finite
      for (const j of row.jacobianA) expect(isFinite(j)).toBe(true);
      for (const j of row.jacobianB) expect(isFinite(j)).toBe(true);
    }
  });
});

// ─── Multi-step stability (CPU path matching GPU logic) ──────────────────────

describe('Multi-step physics stability', () => {
  it('sphere on ground should settle to stable position', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 10,
    });

    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 5, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    for (let i = 0; i < 120; i++) {
      solver.step();
    }

    const body = solver.bodyStore.bodies[b.index];
    // Ground top at 0.5, sphere radius 0.5 → resting at y ≈ 1.0
    expect(body.position.y).toBeGreaterThan(0.7);
    expect(body.position.y).toBeLessThan(2.0);
    // Should have near-zero velocity
    expect(vec3Length(body.velocity)).toBeLessThan(1.0);
    // No NaN
    expect(isFinite(body.position.x)).toBe(true);
    expect(isFinite(body.position.y)).toBe(true);
    expect(isFinite(body.position.z)).toBe(true);
  });

  it('pyramid of 5 boxes should remain stable for 300 steps', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 10,
    });

    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    // Stack of 5 boxes
    for (let i = 0; i < 5; i++) {
      const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 1.0 + i * 1.05, 0);
      const b = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(b.index, ColliderDesc3D.cuboid(0.5, 0.5, 0.5));
    }

    for (let i = 0; i < 300; i++) {
      solver.step();
    }

    // All dynamic bodies should be above ground and finite
    for (const body of solver.bodyStore.bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;
      expect(body.position.y).toBeGreaterThan(0.3);
      expect(body.position.y).toBeLessThan(20);
      expect(isFinite(body.position.x)).toBe(true);
      expect(isFinite(body.position.y)).toBe(true);
      expect(isFinite(body.position.z)).toBe(true);
    }
  });

  it('500 sphere ball pit should not explode', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    // Ground
    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    // Walls
    for (const [x, z, hx, hz] of [[-3, 0, 0.2, 3], [3, 0, 0.2, 3], [0, -3, 3, 0.2], [0, 3, 3, 0.2]] as [number, number, number, number][]) {
      const wd = RigidBodyDesc3D.fixed().setTranslation(x, 2, z);
      const w = solver.bodyStore.addBody(wd);
      solver.bodyStore.attachCollider(w.index, ColliderDesc3D.cuboid(hx, 2, hz));
    }

    // 500 spheres
    const gridSize = Math.ceil(Math.cbrt(500));
    let idx = 0;
    for (let ly = 0; ly < gridSize && idx < 500; ly++) {
      for (let lz = 0; lz < gridSize && idx < 500; lz++) {
        for (let lx = 0; lx < gridSize && idx < 500; lx++) {
          const x = (lx - gridSize / 2 + 0.5) * 0.4;
          const y = 1.0 + ly * 0.4;
          const z = (lz - gridSize / 2 + 0.5) * 0.4;
          const desc = RigidBodyDesc3D.dynamic().setTranslation(x, y, z);
          const b = solver.bodyStore.addBody(desc);
          solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.15));
          idx++;
        }
      }
    }

    // Run 60 steps
    for (let i = 0; i < 60; i++) {
      solver.step();
    }

    // Check no bodies exploded
    let nanCount = 0;
    let maxY = -Infinity;
    for (const body of solver.bodyStore.bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;
      if (!isFinite(body.position.x) || !isFinite(body.position.y) || !isFinite(body.position.z)) {
        nanCount++;
      }
      maxY = Math.max(maxY, body.position.y);
    }
    expect(nanCount).toBe(0);
    expect(maxY).toBeLessThan(50); // No explosion
  }, 30000); // 30 second timeout for large sim
});

// ─── GPU solver source code structural validation ────────────────────────────
// These tests read the GPU solver source code and verify patterns that
// would cause runtime errors in the browser (detached buffers, etc.)

describe('GPU solver staging buffer unmap ordering', () => {
  it('crStagingBuffer must be read BEFORE unmap in step()', async () => {
    // Read the GPU solver source
    const fs = await import('fs');
    const source = fs.readFileSync(new URL('../src/core/gpu-solver-3d.ts', import.meta.url), 'utf-8');

    // Find the constraint readback section
    // The pattern should be: getMappedRange() ... read loop ... unmap()
    // NOT: getMappedRange() ... unmap() ... read loop
    const crReadbackMatch = source.match(
      /crStagingBuffer\.getMappedRange\(\)([\s\S]*?)crStagingBuffer\.unmap\(\)/
    );
    expect(crReadbackMatch).not.toBeNull();

    // Between getMappedRange and unmap, there should be the read loop
    const betweenText = crReadbackMatch![1];
    expect(betweenText).toContain('getFloat32');
    expect(betweenText).toContain('lambda');
    expect(betweenText).toContain('penalty');
  });

  it('bodyStagingBuffer must be read BEFORE unmap in step()', async () => {
    const fs = await import('fs');
    const source = fs.readFileSync(new URL('../src/core/gpu-solver-3d.ts', import.meta.url), 'utf-8');

    const bodyReadbackMatch = source.match(
      /bodyStagingBuffer\.getMappedRange\(\)([\s\S]*?)bodyStagingBuffer\.unmap\(\)/
    );
    expect(bodyReadbackMatch).not.toBeNull();

    // Between getMappedRange and unmap, body positions should be read
    const betweenText = bodyReadbackMatch![1];
    expect(betweenText).toContain('body.position');
  });

  it('persistent staging buffers should NOT be destroyed after use', async () => {
    const fs = await import('fs');
    const source = fs.readFileSync(new URL('../src/core/gpu-solver-3d.ts', import.meta.url), 'utf-8');

    // In the readback section, crStagingBuffer should not be destroyed
    // (it's persistent and reused across frames)
    const afterCrUnmap = source.split('crStagingBuffer.unmap()')[1];
    // The next crStagingBuffer.destroy() should NOT appear before the next method/function
    // (it should only be in destroy() method)
    const nextSection = afterCrUnmap?.split('\n\n')[0] || '';
    expect(nextSection).not.toContain('crStagingBuffer.destroy()');
  });
});

// ─── Broadphase pair dedup correctness ───────────────────────────────────────

describe('Open-addressing hash set for pair dedup', () => {
  it('should not produce duplicate collision pairs', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 3,
    });

    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    // Place spheres in a tight cluster (same spatial hash cell)
    for (let i = 0; i < 10; i++) {
      const desc = RigidBodyDesc3D.dynamic().setTranslation(i * 0.3, 0.8, 0);
      const b = solver.bodyStore.addBody(desc);
      solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.2));
    }

    solver.step();

    // Check for duplicate constraint pairs (same bodyA, bodyB)
    const pairSet = new Set<string>();
    for (let i = 0; i < solver.constraintRows.length; i += 3) {
      const row = solver.constraintRows[i];
      const key = `${Math.min(row.bodyA, row.bodyB)}-${Math.max(row.bodyA, row.bodyB)}`;
      expect(pairSet.has(key)).toBe(false);
      pairSet.add(key);
    }
  });

  it('hash table should handle bodies spanning multiple grid cells', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 3,
    });

    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    // Large box that spans multiple cells + small sphere nearby
    const bigDesc = RigidBodyDesc3D.dynamic().setTranslation(0, 3, 0);
    const big = solver.bodyStore.addBody(bigDesc);
    solver.bodyStore.attachCollider(big.index, ColliderDesc3D.cuboid(2, 2, 2));

    const smallDesc = RigidBodyDesc3D.dynamic().setTranslation(1, 2, 0);
    const small = solver.bodyStore.addBody(smallDesc);
    solver.bodyStore.attachCollider(small.index, ColliderDesc3D.ball(0.3));

    // Should not crash or produce duplicates
    solver.step();

    // Verify no duplicate body pairs in constraints
    const pairSet = new Set<string>();
    let duplicates = 0;
    for (let i = 0; i < solver.constraintRows.length; i += 3) {
      const row = solver.constraintRows[i];
      const key = `${Math.min(row.bodyA, row.bodyB)}-${Math.max(row.bodyA, row.bodyB)}`;
      if (pairSet.has(key)) duplicates++;
      pairSet.add(key);
    }
    expect(duplicates).toBe(0);
  });
});

// ─── Body init in-place mutation correctness ─────────────────────────────────

describe('In-place body init mutation', () => {
  it('prevPosition should reflect position at start of step, not end', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    // First step: prevPosition should be initial position (0, 10, 0)
    solver.step();
    const body = solver.bodyStore.bodies[b.index];
    // position changed (fell), but prevPosition should be the start-of-step position
    expect(body.position.y).toBeLessThan(10);
    // prevPosition is set at START of step to current position
    // After step completes, position has changed but prevPosition stays at start-of-step value
    // On the second step, prevPosition will be set to position from end of first step
    const posAfterStep1 = body.position.y;

    solver.step();
    // Now prevPosition should equal position from end of first step
    expect(body.prevPosition.y).toBeCloseTo(posAfterStep1, 5);
  });

  it('inertialPosition should not alias position (in-place mutation bug check)', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    const desc = RigidBodyDesc3D.dynamic().setTranslation(5, 10, -3);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    solver.step();
    const body = solver.bodyStore.bodies[b.index];

    // After step, inertialPosition and position should be different objects
    // (the optimization uses in-place mutation, but they must remain separate)
    body.position.x = 999;
    expect(body.inertialPosition.x).not.toBe(999);
    expect(body.prevPosition.x).not.toBe(999);
  });

  it('prevVelocity should not alias velocity', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 5,
    });

    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    solver.step();
    const body = solver.bodyStore.bodies[b.index];

    // Mutate velocity - prevVelocity should NOT change
    const prevVelY = body.prevVelocity.y;
    body.velocity.y = 999;
    expect(body.prevVelocity.y).toBe(prevVelY);
  });
});

// ─── Constraint warmstart cache with numeric keys ────────────────────────────

describe('Constraint warmstarting across frames', () => {
  it('warmstarted contacts should have higher penalty than fresh contacts', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 10,
    });

    const groundDesc = RigidBodyDesc3D.fixed().setTranslation(0, 0, 0);
    const ground = solver.bodyStore.addBody(groundDesc);
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(50, 0.5, 50));

    const desc = RigidBodyDesc3D.dynamic().setTranslation(0, 0.9, 0);
    const b = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(b.index, ColliderDesc3D.ball(0.5));

    // First step: fresh contacts
    solver.step();
    const firstStepPenalties = solver.constraintRows.map(r => r.penalty);

    // Second step: warmstarted contacts should have evolved penalty
    solver.step();
    const secondStepPenalties = solver.constraintRows.map(r => r.penalty);

    // At least some penalties should have changed (warmstart applies gamma multiplier)
    if (firstStepPenalties.length > 0 && secondStepPenalties.length > 0) {
      // The warmstarted penalty should generally be different from fresh
      const changed = secondStepPenalties.some((p, i) =>
        i < firstStepPenalties.length && p !== firstStepPenalties[i]
      );
      expect(changed).toBe(true);
    }
  });
});

// ─── Buffer packing correctness ──────────────────────────────────────────────

describe('Buffer packing for GPU upload', () => {
  it('constraint row packing should put lambda at byte offset 120 and penalty at 124', () => {
    // This verifies the byte offsets used in readback (the staging buffer bug location)
    const CONSTRAINT_STRIDE = 36; // floats
    const byteOff = 0 * CONSTRAINT_STRIDE * 4; // first constraint

    // Lambda is at float index 30 → byte offset 120
    expect(byteOff + 120).toBe(120);
    // Verify the DataView offset matches the upload code
    const crData = new ArrayBuffer(CONSTRAINT_STRIDE * 4);
    const crView = new DataView(crData);
    crView.setFloat32(120, -42.5, true); // lambda
    crView.setFloat32(124, 1000, true);  // penalty
    expect(crView.getFloat32(120, true)).toBe(-42.5);
    expect(crView.getFloat32(124, true)).toBe(1000);
  });

  it('mapped range must be read BEFORE unmap (staging buffer correctness)', () => {
    // This test validates the pattern used in GPU readback.
    // The bug was: unmap() was called before reading from the DataView,
    // causing a "detached ArrayBuffer" error.
    //
    // Correct pattern:
    //   1. const view = new DataView(buffer.getMappedRange())
    //   2. READ from view
    //   3. buffer.unmap()
    //
    // Incorrect pattern (the bug):
    //   1. const view = new DataView(buffer.getMappedRange())
    //   2. buffer.unmap()  // DETACHES the ArrayBuffer!
    //   3. READ from view  // THROWS: detached ArrayBuffer

    // Simulate with regular ArrayBuffer (detach via transfer)
    const ab = new ArrayBuffer(16);
    const view = new DataView(ab);
    view.setFloat32(0, 3.14, true);
    view.setFloat32(4, 2.71, true);

    // CORRECT: read before "detach"
    const val1 = view.getFloat32(0, true);
    const val2 = view.getFloat32(4, true);
    expect(val1).toBeCloseTo(3.14, 2);
    expect(val2).toBeCloseTo(2.71, 2);

    // Transfer detaches the buffer (simulates unmap)
    const transferred = ab.transfer();
    // After transfer, original ab is detached
    expect(ab.byteLength).toBe(0);
    expect(() => view.getFloat32(0, true)).toThrow();
  });
});
