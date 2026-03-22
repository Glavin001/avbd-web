import { describe, it, expect } from 'vitest';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';
import { AVBDSolver3D } from '../src/core/solver-3d.js';
import { RigidBodyDesc3D, ColliderDesc3D } from '../src/core/rigid-body-3d.js';
import { solveLDL3, mat3Zero, mat3Set } from '../src/core/math.js';

describe('Edge cases — 2D solver', () => {
  it('should handle empty world gracefully', () => {
    const solver = new AVBDSolver2D();
    expect(() => solver.step()).not.toThrow();
  });

  it('should handle single body with no colliders', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: -9.81 } });
    const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 5);
    solver.bodyStore.addBody(desc);
    // Body has zero mass (no collider), should not crash
    expect(() => solver.step()).not.toThrow();
  });

  it('should handle body at origin', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: 0 } });
    const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 0);
    const handle = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

    for (let i = 0; i < 10; i++) solver.step();

    const body = solver.bodyStore.getBody(handle);
    expect(body.position.x).toBeCloseTo(0);
    expect(body.position.y).toBeCloseTo(0);
  });

  it('should handle very small bodies', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: -9.81 } });
    const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 5);
    const handle = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.01, 0.01));
    expect(() => {
      for (let i = 0; i < 30; i++) solver.step();
    }).not.toThrow();
  });

  it('should handle very large bodies', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: -9.81 } });
    const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 50);
    const handle = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(10, 10));
    expect(() => {
      for (let i = 0; i < 30; i++) solver.step();
    }).not.toThrow();
  });

  it('should handle high velocity', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: 0 } });
    const desc = RigidBodyDesc2D.dynamic()
      .setTranslation(0, 0)
      .setLinvel(1000, 1000);
    const handle = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));
    expect(() => {
      for (let i = 0; i < 10; i++) solver.step();
    }).not.toThrow();
  });

  it('should not produce NaN positions', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: -9.81 }, iterations: 10 });

    // Ground
    const ground = solver.bodyStore.addBody(RigidBodyDesc2D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc2D.cuboid(10, 0.5));

    // Stack of boxes
    for (let i = 0; i < 5; i++) {
      const h = solver.bodyStore.addBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 1 + i * 1.1)
      );
      solver.bodyStore.attachCollider(h.index, ColliderDesc2D.cuboid(0.5, 0.5));
    }

    for (let i = 0; i < 300; i++) {
      solver.step();
      for (const body of solver.bodyStore.bodies) {
        expect(isFinite(body.position.x)).toBe(true);
        expect(isFinite(body.position.y)).toBe(true);
        expect(isFinite(body.angle)).toBe(true);
      }
    }
  });

  it('should handle zero gravity', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: 0 } });
    const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 5);
    const handle = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

    for (let i = 0; i < 30; i++) solver.step();

    const body = solver.bodyStore.getBody(handle);
    expect(body.position.y).toBeCloseTo(5); // Should not move
  });

  it('should handle negative gravity (upward)', () => {
    const solver = new AVBDSolver2D({ gravity: { x: 0, y: 9.81 } });
    const desc = RigidBodyDesc2D.dynamic().setTranslation(0, 0);
    const handle = solver.bodyStore.addBody(desc);
    solver.bodyStore.attachCollider(handle.index, ColliderDesc2D.cuboid(0.5, 0.5));

    for (let i = 0; i < 30; i++) solver.step();

    const body = solver.bodyStore.getBody(handle);
    expect(body.position.y).toBeGreaterThan(0); // Should float upward
  });
});

describe('Edge cases — 3D solver', () => {
  it('should handle empty world', () => {
    const solver = new AVBDSolver3D();
    expect(() => solver.step()).not.toThrow();
  });

  it('should handle free fall in 3D', () => {
    const solver = new AVBDSolver3D({ gravity: { x: 0, y: -9.81, z: 0 } });
    const handle = solver.bodyStore.addBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 10, 0)
    );
    solver.bodyStore.attachCollider(handle.index, ColliderDesc3D.cuboid(0.5, 0.5, 0.5));

    for (let i = 0; i < 30; i++) solver.step();

    const body = solver.bodyStore.getBody(handle);
    expect(body.position.y).toBeLessThan(10);
    expect(isFinite(body.position.x)).toBe(true);
    expect(isFinite(body.position.z)).toBe(true);
  });

  it('should not produce NaN in 3D stacking', () => {
    const solver = new AVBDSolver3D({
      gravity: { x: 0, y: -9.81, z: 0 },
      iterations: 10,
    });

    const ground = solver.bodyStore.addBody(RigidBodyDesc3D.fixed());
    solver.bodyStore.attachCollider(ground.index, ColliderDesc3D.cuboid(10, 0.5, 10));

    for (let i = 0; i < 3; i++) {
      const h = solver.bodyStore.addBody(
        RigidBodyDesc3D.dynamic().setTranslation(0, 1.5 + i * 1.1, 0)
      );
      solver.bodyStore.attachCollider(h.index, ColliderDesc3D.cuboid(0.5, 0.5, 0.5));
    }

    for (let step = 0; step < 120; step++) {
      solver.step();
      for (const body of solver.bodyStore.bodies) {
        expect(isFinite(body.position.x)).toBe(true);
        expect(isFinite(body.position.y)).toBe(true);
        expect(isFinite(body.position.z)).toBe(true);
      }
    }
  });
});

describe('Edge cases — LDL solver', () => {
  it('should handle very large diagonal values', () => {
    const A = mat3Zero();
    mat3Set(A, 0, 0, 1e12);
    mat3Set(A, 1, 1, 1e12);
    mat3Set(A, 2, 2, 1e12);
    const x = solveLDL3(A, [1e12, 1e12, 1e12]);
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(1);
    expect(x[2]).toBeCloseTo(1);
  });

  it('should handle asymmetric RHS', () => {
    const A = mat3Zero();
    mat3Set(A, 0, 0, 1);
    mat3Set(A, 1, 1, 1);
    mat3Set(A, 2, 2, 1);
    const x = solveLDL3(A, [1, -1, 0.5]);
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(-1);
    expect(x[2]).toBeCloseTo(0.5);
  });
});
