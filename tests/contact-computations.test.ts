/**
 * Exhaustive unit tests for contact constraint atomic computations.
 * Tests every function in src/constraints/contact.ts and 3D contact creation.
 */
import { describe, it, expect } from 'vitest';
import { createContactConstraintRows, updateFrictionBounds } from '../src/constraints/contact.js';
import { createDefaultRow, type ConstraintRow } from '../src/constraints/constraint.js';
import { ForceType, COLLISION_MARGIN } from '../src/core/types.js';
import type { Vec2, ContactManifold2D } from '../src/core/types.js';
import { AVBDSolver2D } from '../src/core/solver.js';
import { RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';
import { World3D, RigidBodyDesc3D, ColliderDesc3D } from '../src/3d/index.js';
import { vec2Cross, vec2Dot } from '../src/core/math.js';

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeBody2D(x: number, y: number, opts: {
  friction?: number; restitution?: number; vx?: number; vy?: number; omega?: number;
  index?: number;
} = {}): any {
  return {
    position: { x, y }, angle: 0,
    velocity: { x: opts.vx ?? 0, y: opts.vy ?? 0 },
    angularVelocity: opts.omega ?? 0,
    friction: opts.friction ?? 0.5,
    restitution: opts.restitution ?? 0,
    index: opts.index ?? 0, type: 0,
  };
}

function makeManifold(bodyA: number, bodyB: number, normal: Vec2,
  contacts: { position: Vec2; depth: number }[]): ContactManifold2D {
  return { bodyA, bodyB, normal, contacts } as ContactManifold2D;
}

// ─── Jacobian Verification ──────────────────────────────────────────────────
// computeContactJacobian is private but we can verify its output through
// createContactConstraintRows which uses it for jacobianA and jacobianB.

describe('Contact Jacobian computation', () => {
  it('contact at body center: zero angular Jacobian', () => {
    // Body at (0,0), contact at (0,0), normal (0,1)
    // r = (0,0), r×n = 0*1 - 0*0 = 0
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    const normalRow = rows[0];
    // J_A = [nx, ny, r×n] = [0, 1, 0] (contact at body center → zero torque)
    expect(normalRow.jacobianA[0]).toBeCloseTo(0);
    expect(normalRow.jacobianA[1]).toBeCloseTo(1);
    expect(normalRow.jacobianA[2]).toBeCloseTo(0);
  });

  it('contact offset right: positive angular Jacobian', () => {
    // Body at (0,0), contact at (1,0), normal (0,1)
    // r = (1,0), r×n = 1*1 - 0*0 = 1
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 1, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].jacobianA[0]).toBeCloseTo(0);
    expect(rows[0].jacobianA[1]).toBeCloseTo(1);
    expect(rows[0].jacobianA[2]).toBeCloseTo(1); // r×n = 1
  });

  it('contact offset up: negative angular Jacobian', () => {
    // Body at (0,0), contact at (0,1), normal (1,0)
    // r = (0,1), r×n = 0*0 - 1*1 = -1
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(1, 0, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 1, y: 0 }, [
      { position: { x: 0, y: 1 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].jacobianA[0]).toBeCloseTo(1);
    expect(rows[0].jacobianA[1]).toBeCloseTo(0);
    expect(rows[0].jacobianA[2]).toBeCloseTo(-1); // r×n = -1
  });

  it('jacobianB is negated from its own computation', () => {
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(2, 0, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 1, y: 0 }, [
      { position: { x: 1, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    // J_B linear should be negative of normal
    expect(rows[0].jacobianB[0]).toBeCloseTo(-1);
    expect(rows[0].jacobianB[1]).toBeCloseTo(0);
    // J_B angular should be negated: -(rB × n) where rB = (1,0)-(2,0) = (-1,0)
    // rB × n = (-1)*0 - 0*1 = 0, negated = 0
    // Actually rB × (1,0) = (-1)*0 - 0*1 = 0
    expect(rows[0].jacobianB[2]).toBeCloseTo(0);
  });

  it('large offset produces large angular Jacobian', () => {
    // Body at (0,0), contact at (5,0), normal (0,1)
    // r = (5,0), r×n = 5*1 - 0*0 = 5
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 5, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].jacobianA[2]).toBeCloseTo(5);
  });
});

// ─── Hessian Diagonal Verification ──────────────────────────────────────────

describe('Contact Hessian diagonal (geometric stiffness)', () => {
  it('normal hessianDiag = [0, 0, -(rA·n)]', () => {
    // Body at (0,0), contact at (1,0), normal (1,0)
    // rA = (1,0), rA·n = 1*1 + 0*0 = 1
    // hessianDiagA[2] = -1
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(2, 0, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 1, y: 0 }, [
      { position: { x: 1, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].hessianDiagA[0]).toBeCloseTo(0);
    expect(rows[0].hessianDiagA[1]).toBeCloseTo(0);
    expect(rows[0].hessianDiagA[2]).toBeCloseTo(-1); // -(rA·n) = -1
  });

  it('hessianDiagB = [0, 0, -(rB·n)]', () => {
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(3, 0, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 1, y: 0 }, [
      { position: { x: 1, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    // rB = contact - bodyB = (1,0)-(3,0) = (-2,0), rB·n = (-2)*1+0*0 = -2
    expect(rows[0].hessianDiagB[2]).toBeCloseTo(2); // -(rB·n) = -(-2) = 2
  });

  it('friction hessianDiag = [0, 0, -(rA·t)] where t = perp(n)', () => {
    // normal = (0,1), tangent = (-1, 0) [perpendicular: {-ny, nx}]
    // wait — tangent = { x: -n.y, y: n.x } = { x: -1, y: 0 }
    // rA = (2,0), rA·t = 2*(-1) + 0*0 = -2
    // friction hessianDiagA[2] = -(rA·t) = -(-2) = 2
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 2, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    const fricRow = rows[1]; // friction row
    // tangent = (-1, 0), rA = (2,0), rA·t = -2, hessianDiag = -(-2) = 2
    expect(fricRow.hessianDiagA[2]).toBeCloseTo(2);
  });

  it('contact at center: hessianDiag = [0,0,0]', () => {
    const bodyA = makeBody2D(1, 1, { index: 0 });
    const bodyB = makeBody2D(1, 2, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 1, y: 1 }, depth: 0.1 }, // contact at bodyA center
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    // rA = (0,0), so rA·n = 0
    expect(rows[0].hessianDiagA[2]).toBeCloseTo(0);
    expect(rows[1].hessianDiagA[2]).toBeCloseTo(0); // friction too
  });

  it('r perpendicular to n: normal hessianDiag = 0', () => {
    // rA = (1,0), n = (0,1), rA·n = 0
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 1, y: 0 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].hessianDiagA[2]).toBeCloseTo(0); // -(rA·n) = 0
    // But friction hessianDiag should be non-zero:
    // tangent = (-1,0), rA·t = 1*(-1)+0*0 = -1
    expect(rows[1].hessianDiagA[2]).toBeCloseTo(1); // -(rA·t) = 1
  });
});

// ─── Restitution ────────────────────────────────────────────────────────────

describe('Contact restitution', () => {
  it('no restitution: c0 equals c', () => {
    const bodyA = makeBody2D(0, 0, { index: 0, restitution: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1, restitution: 0 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].c0).toBeCloseTo(rows[0].c);
  });

  it('restitution with slow approach: c0 equals c (threshold not met)', () => {
    // vn > -0.5 → no bounce
    const bodyA = makeBody2D(0, 0, { index: 0, restitution: 0.5, vy: -0.1 });
    const bodyB = makeBody2D(0, 1, { index: 1, restitution: 0.5 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].c0).toBeCloseTo(rows[0].c);
  });

  it('restitution with fast approach: c0 biased for bounce', () => {
    // vn < -0.5 → bounce applied
    // bodyA moving down fast: vy = -5
    // relative normal velocity = vA·n - vB·n (projected onto normal (0,1))
    // vA at contact = (0, -5), vB = (0, 0), relVel·n = -5
    const bodyA = makeBody2D(0, 0, { index: 0, restitution: 0.8, vy: -5 });
    const bodyB = makeBody2D(0, 1, { index: 1, restitution: 0 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0 }, depth: 0.1 },
    ]);
    const dt = 1 / 60;
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100, Infinity, dt);
    // restitution = max(0.8, 0) = 0.8, vn ≈ -5 < -0.5
    // c0 = c + e * vn * dt = c + 0.8 * (-5) * (1/60)
    const expectedBias = 0.8 * (-5) * dt;
    expect(rows[0].c0).toBeCloseTo(rows[0].c + expectedBias, 3);
  });
});

// ─── Friction Coefficient ───────────────────────────────────────────────────

describe('Contact friction coefficient', () => {
  it('equal friction: geometric mean', () => {
    const bodyA = makeBody2D(0, 0, { index: 0, friction: 0.5 });
    const bodyB = makeBody2D(0, 1, { index: 1, friction: 0.5 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    const mu = Math.sqrt(0.5 * 0.5);
    const expectedBound = mu * 100 * 0.1;
    expect(rows[1].fmax).toBeCloseTo(expectedBound);
    expect(rows[1].fmin).toBeCloseTo(-expectedBound);
  });

  it('zero friction one side: mu = 0', () => {
    const bodyA = makeBody2D(0, 0, { index: 0, friction: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1, friction: 1.0 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[1].fmax).toBeCloseTo(0);
    expect(rows[1].fmin).toBeCloseTo(0);
  });

  it('asymmetric friction: geometric mean', () => {
    const bodyA = makeBody2D(0, 0, { index: 0, friction: 0.2 });
    const bodyB = makeBody2D(0, 1, { index: 1, friction: 0.8 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    const mu = Math.sqrt(0.2 * 0.8);
    const expectedBound = mu * 100 * 0.1;
    expect(rows[1].fmax).toBeCloseTo(expectedBound);
  });
});

// ─── Constraint Bounds ──────────────────────────────────────────────────────

describe('Contact constraint bounds', () => {
  it('normal row: fmin=-Infinity, fmax=0 (compressive only)', () => {
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].fmin).toBe(-Infinity);
    expect(rows[0].fmax).toBe(0);
  });

  it('friction row: symmetric bounds from Coulomb estimate', () => {
    const bodyA = makeBody2D(0, 0, { index: 0, friction: 0.5 });
    const bodyB = makeBody2D(0, 1, { index: 1, friction: 0.5 });
    const depth = 0.2;
    const penaltyMin = 200;
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, penaltyMin);
    const mu = 0.5;
    const expected = mu * penaltyMin * depth;
    expect(rows[1].fmax).toBeCloseTo(expected);
    expect(rows[1].fmin).toBeCloseTo(-expected);
  });

  it('constraint value includes collision margin', () => {
    const depth = 0.1;
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows[0].c).toBeCloseTo(-depth + COLLISION_MARGIN);
  });
});

// ─── updateFrictionBounds ───────────────────────────────────────────────────

describe('updateFrictionBounds', () => {
  it('should set friction bounds from normal lambda', () => {
    const normalRow = createDefaultRow();
    normalRow.lambda = -100; // Compressive force
    const frictionRow = createDefaultRow();
    const bodyA = makeBody2D(0, 0, { friction: 0.5 });
    const bodyB = makeBody2D(0, 1, { friction: 0.5 });

    updateFrictionBounds(normalRow, frictionRow, bodyA, bodyB);

    const mu = 0.5;
    expect(frictionRow.fmax).toBeCloseTo(mu * 100);
    expect(frictionRow.fmin).toBeCloseTo(-mu * 100);
  });

  it('zero normal force → zero friction bounds', () => {
    const normalRow = createDefaultRow();
    normalRow.lambda = 0;
    const frictionRow = createDefaultRow();
    const bodyA = makeBody2D(0, 0, { friction: 0.5 });
    const bodyB = makeBody2D(0, 1, { friction: 0.5 });

    updateFrictionBounds(normalRow, frictionRow, bodyA, bodyB);

    expect(frictionRow.fmax).toBeCloseTo(0);
    expect(frictionRow.fmin).toBeCloseTo(0);
  });

  it('different friction coefficients', () => {
    const normalRow = createDefaultRow();
    normalRow.lambda = -50;
    const frictionRow = createDefaultRow();
    const bodyA = makeBody2D(0, 0, { friction: 0.3 });
    const bodyB = makeBody2D(0, 1, { friction: 0.7 });

    updateFrictionBounds(normalRow, frictionRow, bodyA, bodyB);

    const mu = Math.sqrt(0.3 * 0.7);
    expect(frictionRow.fmax).toBeCloseTo(mu * 50);
  });
});

// ─── Row Count ──────────────────────────────────────────────────────────────

describe('Contact row count', () => {
  it('1 contact → 2 rows (normal + friction)', () => {
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows.length).toBe(2);
  });

  it('3 contacts → 6 rows', () => {
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: -1, y: 0.5 }, depth: 0.1 },
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
      { position: { x: 1, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    expect(rows.length).toBe(6);
  });

  it('all rows have ForceType.Contact', () => {
    const bodyA = makeBody2D(0, 0, { index: 0 });
    const bodyB = makeBody2D(0, 1, { index: 1 });
    const manifold = makeManifold(0, 1, { x: 0, y: 1 }, [
      { position: { x: 0, y: 0.5 }, depth: 0.1 },
    ]);
    const rows = createContactConstraintRows(manifold, bodyA, bodyB, 100);
    for (const row of rows) {
      expect(row.type).toBe(ForceType.Contact);
    }
  });
});

// ─── 3D Contact Constraint Creation ─────────────────────────────────────────

describe('3D contact constraint creation', () => {
  it('should create 3 rows per contact point (normal + 2 friction)', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 1, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 0.9, 0),
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), body);

    world.step();

    // Access internal solver to check constraint rows
    const solver = (world as any).solver || (world as any).cpuSolver;
    if (solver && solver.constraintRows) {
      const contactRows = solver.constraintRows.filter((r: any) => r.active && r.type === ForceType.Contact);
      // Each contact point generates 3 rows (normal + 2 friction)
      expect(contactRows.length % 3).toBe(0);
      expect(contactRows.length).toBeGreaterThan(0);
    }
  });

  it('3D hessianDiag should have 6 components', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 1, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 0.9, 0),
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), body);

    world.step();

    const solver = (world as any).solver || (world as any).cpuSolver;
    if (solver && solver.constraintRows) {
      for (const row of solver.constraintRows) {
        if (!row.active || row.type !== ForceType.Contact) continue;
        expect(row.hessianDiagA.length).toBe(6);
        expect(row.hessianDiagB.length).toBe(6);
        // Linear components should be zero for contacts
        expect(row.hessianDiagA[0]).toBeCloseTo(0);
        expect(row.hessianDiagA[1]).toBeCloseTo(0);
        expect(row.hessianDiagA[2]).toBeCloseTo(0);
        // All values should be finite
        for (let i = 0; i < 6; i++) {
          expect(isFinite(row.hessianDiagA[i])).toBe(true);
          expect(isFinite(row.hessianDiagB[i])).toBe(true);
        }
        break; // Just check first row
      }
    }
  });

  it('3D Jacobian should have 6 components', () => {
    const world = new World3D({ x: 0, y: -9.81, z: 0 }, {
      iterations: 1, useCPU: true,
    });
    world.createCollider(ColliderDesc3D.cuboid(10, 0.5, 10).setFriction(0.5));
    const body = world.createRigidBody(
      RigidBodyDesc3D.dynamic().setTranslation(0, 0.9, 0),
    );
    world.createCollider(ColliderDesc3D.cuboid(0.5, 0.5, 0.5).setFriction(0.5), body);

    world.step();

    const solver = (world as any).solver || (world as any).cpuSolver;
    if (solver && solver.constraintRows) {
      for (const row of solver.constraintRows) {
        if (!row.active) continue;
        expect(row.jacobianA.length).toBe(6);
        expect(row.jacobianB.length).toBe(6);
        break;
      }
    }
  });
});
