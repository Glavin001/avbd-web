/**
 * AVBD (Augmented Vertex Block Descent) solver for 2D rigid bodies.
 * This is the CPU reference implementation, ported from avbd-demo2d/source/solver.cpp.
 *
 * The solver loop per timestep:
 * 1. Broadphase collision detection
 * 2. Initialize & warmstart constraints (decay penalty/lambda)
 * 3. Initialize bodies (compute inertial positions)
 * 4. Main solver loop:
 *    a. Primal update (per body): build 3x3 system, solve via LDL^T
 *    b. Dual update (per constraint): update lambda, ramp penalty
 * 5. Velocity recovery (BDF1)
 */

import type { Vec2, SolverConfig, ContactManifold2D } from './types.js';
import { RigidBodyType, DEFAULT_SOLVER_CONFIG_2D } from './types.js';
import { ForceType } from './types.js';
import type { Body2D } from './rigid-body.js';
import { BodyStore2D } from './rigid-body.js';
import type { ConstraintRow } from '../constraints/constraint.js';
import { ConstraintStore } from '../constraints/constraint.js';
import { createContactConstraintRows, updateFrictionBounds } from '../constraints/contact.js';
import { collide2D } from '../2d/collision-sat.js';
import { aabb2DOverlap } from './math.js';
import {
  mat3Zero, mat3Identity, mat3Scale, mat3Get, mat3Set, mat3Add, mat3OuterProduct,
  solveLDL3, diagonalGeometricStiffness3,
  vec2Add, vec2Sub, vec2Scale, vec2Length,
} from './math.js';
import { computeGraphColoring, type ColorGroup } from './graph-coloring.js';

// ─── Broadphase ─────────────────────────────────────────────────────────────

export interface IgnoreCollisionPair {
  bodyA: number;
  bodyB: number;
}

function broadphase2D(
  bodyStore: BodyStore2D,
  ignorePairs: Set<string>,
): ContactManifold2D[] {
  const manifolds: ContactManifold2D[] = [];
  const bodies = bodyStore.bodies;
  const n = bodies.length;

  for (let i = 0; i < n; i++) {
    const a = bodies[i];
    const aabbA = bodyStore.getAABB(a);

    for (let j = i + 1; j < n; j++) {
      const b = bodies[j];

      // Skip pairs where neither body is dynamic
      if (a.type !== RigidBodyType.Dynamic && b.type !== RigidBodyType.Dynamic) continue;

      // Check ignore list
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (ignorePairs.has(key)) continue;

      // AABB overlap test
      const aabbB = bodyStore.getAABB(b);
      if (!aabb2DOverlap(aabbA, aabbB)) continue;

      // Narrowphase
      const manifold = collide2D(a, b);
      if (manifold) {
        manifolds.push(manifold);
      }
    }
  }

  return manifolds;
}

// ─── AVBD Solver ────────────────────────────────────────────────────────────

export class AVBDSolver2D {
  config: SolverConfig;
  bodyStore: BodyStore2D;
  constraintStore: ConstraintStore;
  ignorePairs: Set<string> = new Set();

  /** Persistent joint constraint indices (not cleared each frame) */
  jointConstraintIndices: number[] = [];

  /** Graph coloring for parallel dispatch (updated each step) */
  colorGroups: ColorGroup[] = [];

  constructor(config: Partial<SolverConfig> = {}) {
    this.config = { ...DEFAULT_SOLVER_CONFIG_2D, ...config };
    this.bodyStore = new BodyStore2D();
    this.constraintStore = new ConstraintStore();
  }

  /**
   * Perform one physics timestep.
   * This is the main AVBD solver loop from avbd-demo2d/source/solver.cpp.
   */
  step(): void {
    const { config, bodyStore, constraintStore } = this;
    const dt = config.dt;
    const gravity = config.gravity as Vec2;
    const bodies = bodyStore.bodies;

    if (bodies.length === 0) return;

    // ─── 1. Broadphase & Narrowphase ──────────────────────────────
    // Clear contact constraints (keep joints)
    constraintStore.clearContacts();

    const manifolds = broadphase2D(bodyStore, this.ignorePairs);

    // Create contact constraint rows
    for (const manifold of manifolds) {
      const bodyA = bodies[manifold.bodyA];
      const bodyB = bodies[manifold.bodyB];
      const rows = createContactConstraintRows(
        manifold, bodyA, bodyB,
        config.penaltyMin, Infinity,
      );
      constraintStore.addRows(rows);
    }

    // Apply cached warmstart values to new contacts
    constraintStore.warmstartContacts();

    // ─── 2. Initialize & Warmstart Constraints ────────────────────
    for (const row of constraintStore.rows) {
      if (!row.active) continue;

      // Warmstart: decay penalty and lambda
      row.penalty *= config.gamma;
      row.penalty = Math.max(config.penaltyMin, Math.min(config.penaltyMax, row.penalty));

      // Cap penalty at material stiffness
      if (row.penalty > row.stiffness) {
        row.penalty = row.stiffness;
      }

      if (config.postStabilize) {
        // With post-stabilization, lambda is reused fully
      } else {
        row.lambda *= config.alpha * config.gamma;
      }
    }

    // ─── 2b. Graph Coloring (for GPU dispatch ordering) ───────────
    const constraintPairs = constraintStore.getConstraintPairs();
    const fixedBodies = new Set<number>();
    for (const body of bodies) {
      if (body.type === RigidBodyType.Fixed) fixedBodies.add(body.index);
    }
    this.colorGroups = computeGraphColoring(
      bodies.length, constraintPairs, fixedBodies,
    );

    // ─── 3. Initialize Bodies ─────────────────────────────────────
    const MAX_ANGULAR_VELOCITY = 50; // Reference clamps to [-50, 50]
    const gravMag = vec2Length(gravity);

    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;

      // Clamp angular velocity (reference: clamp(omega, -50, 50))
      body.angularVelocity = Math.max(-MAX_ANGULAR_VELOCITY,
        Math.min(MAX_ANGULAR_VELOCITY, body.angularVelocity));

      // Save initial position
      body.prevPosition = { ...body.position };
      body.prevAngle = body.angle;

      // Adaptive gravity weighting (from reference solver.cpp)
      // Bodies under support get less gravity in the inertial estimate
      // accelWeight = clamp(accel_along_gravity / |gravity|, 0, 1)
      let gravWeight = 1;
      if (gravMag > 0) {
        const dvx = body.velocity.x - body.prevVelocity.x;
        const dvy = body.velocity.y - body.prevVelocity.y;
        const dvMag = Math.sqrt(dvx * dvx + dvy * dvy);
        // Only apply adaptive weighting when there's meaningful velocity change
        if (dvMag > 0.01) {
          const gravDir = { x: gravity.x / gravMag, y: gravity.y / gravMag };
          const accelInGravDir = (dvx * gravDir.x + dvy * gravDir.y) / dt;
          // sign(gravity) * accel — positive when accelerating with gravity
          gravWeight = Math.max(0, Math.min(1, accelInGravDir / gravMag));
        }
      }

      // Apply velocity damping
      let vx = body.velocity.x;
      let vy = body.velocity.y;
      let omega = body.angularVelocity;

      if (body.linearDamping > 0) {
        const dampFactor = 1 / (1 + body.linearDamping * dt);
        vx *= dampFactor;
        vy *= dampFactor;
      }
      if (body.angularDamping > 0) {
        const dampFactor = 1 / (1 + body.angularDamping * dt);
        omega *= dampFactor;
      }

      // Compute inertial target with adaptive gravity
      body.inertialPosition = {
        x: body.position.x + vx * dt + gravity.x * body.gravityScale * gravWeight * dt * dt,
        y: body.position.y + vy * dt + gravity.y * body.gravityScale * gravWeight * dt * dt,
      };
      body.inertialAngle = body.angle + omega * dt;

      // Save velocity for next frame's adaptive gravity
      body.prevVelocity = { ...body.velocity };
    }

    // ─── 4. Main Solver Loop ──────────────────────────────────────
    const totalIterations = config.postStabilize ? config.iterations + 1 : config.iterations;

    for (let iter = 0; iter < totalIterations; iter++) {
      const isStabilization = config.postStabilize && iter === totalIterations - 1;

      // ─── 4a. Primal Update (per body) ────────────────────────
      for (const body of bodies) {
        if (body.type !== RigidBodyType.Dynamic) continue;

        this.primalUpdate(body, isStabilization, dt);
      }

      // ─── 4b. Dual Update (per constraint) ────────────────────
      // Skip dual update on stabilization iteration
      if (!isStabilization) {
        for (let i = 0; i < constraintStore.rows.length; i++) {
          const row = constraintStore.rows[i];
          if (!row.active || row.broken) continue;

          this.dualUpdate(row, i, dt);
        }
      }

      // ─── 4c. Velocity Recovery (at iteration N-1) ────────────
      if (iter === config.iterations - 1) {
        for (const body of bodies) {
          if (body.type !== RigidBodyType.Dynamic) continue;

          body.velocity = {
            x: (body.position.x - body.prevPosition.x) / dt,
            y: (body.position.y - body.prevPosition.y) / dt,
          };
          body.angularVelocity = (body.angle - body.prevAngle) / dt;
        }
      }
    }
  }

  /**
   * Primal update for a single body.
   * Builds the 3x3 SPD system and solves via LDL^T.
   * From avbd-demo2d: solver.cpp lines ~150-200
   */
  private primalUpdate(body: Body2D, isStabilization: boolean, dt: number): void {
    const dt2 = dt * dt;
    const rows = this.constraintStore.rows;

    // LHS starts as M/dt^2 (mass matrix)
    const lhs = mat3Zero();
    mat3Set(lhs, 0, 0, body.mass / dt2);
    mat3Set(lhs, 1, 1, body.mass / dt2);
    mat3Set(lhs, 2, 2, body.inertia / dt2);

    // RHS starts as M/dt^2 * (x - x_inertial)
    const dx = body.position.x - body.inertialPosition.x;
    const dy = body.position.y - body.inertialPosition.y;
    const dtheta = body.angle - body.inertialAngle;
    const rhs: [number, number, number] = [
      body.mass / dt2 * dx,
      body.mass / dt2 * dy,
      body.inertia / dt2 * dtheta,
    ];

    // Accumulate contributions from all constraint rows involving this body
    for (const row of rows) {
      if (!row.active || row.broken) continue;

      let J: [number, number, number] | null = null;

      if (row.bodyA === body.index) {
        J = row.jacobianA;
      } else if (row.bodyB === body.index) {
        J = row.jacobianB;
      } else {
        continue;
      }

      // Evaluate constraint (Taylor-series linearization)
      // C = C0*(1-alpha) + J_A*dp_A + J_B*dp_B
      let cEval = row.c0;
      if (row.bodyA >= 0) {
        const bA = this.bodyStore.bodies[row.bodyA];
        cEval += row.jacobianA[0] * (bA.position.x - bA.prevPosition.x)
              + row.jacobianA[1] * (bA.position.y - bA.prevPosition.y)
              + row.jacobianA[2] * (bA.angle - bA.prevAngle);
      }
      if (row.bodyB >= 0) {
        const bB = this.bodyStore.bodies[row.bodyB];
        cEval += row.jacobianB[0] * (bB.position.x - bB.prevPosition.x)
              + row.jacobianB[1] * (bB.position.y - bB.prevPosition.y)
              + row.jacobianB[2] * (bB.angle - bB.prevAngle);
      }

      // For soft constraints (finite stiffness), zero lambda in primal update
      // Only hard constraints (infinite stiffness) use warmstarted lambda
      // Reference: solver.cpp "lambda = isinf(stiffness[i]) ? force->lambda[i] : 0.0f"
      const lambdaForPrimal = isFinite(row.stiffness) ? 0 : row.lambda;

      // Clamped force: f = clamp(penalty * C + lambda, fmin, fmax)
      let f = row.penalty * cEval + lambdaForPrimal;
      f = Math.max(row.fmin, Math.min(row.fmax, f));

      // Accumulate RHS: += J * f
      rhs[0] += J[0] * f;
      rhs[1] += J[1] * f;
      rhs[2] += J[2] * f;

      // Accumulate LHS: += J * J^T * penalty
      const JJT = mat3OuterProduct(J);
      const penaltyContrib = mat3Scale(JJT, row.penalty);
      const lhsNew = mat3Add(lhs, penaltyContrib);
      for (let k = 0; k < 9; k++) lhs[k] = lhsNew[k];

      // Geometric stiffness (diagonal lumping)
      const hDiag = row.bodyA === body.index ? row.hessianDiagA : row.hessianDiagB;
      const G = diagonalGeometricStiffness3(hDiag, Math.abs(f));
      const lhsG = mat3Add(lhs, G);
      for (let k = 0; k < 9; k++) lhs[k] = lhsG[k];
    }

    // Check if LHS is valid (non-zero diagonal)
    if (mat3Get(lhs, 0, 0) <= 0 || mat3Get(lhs, 1, 1) <= 0 || mat3Get(lhs, 2, 2) <= 0) {
      return; // Degenerate system, skip
    }

    // Solve: LHS * delta = RHS
    const delta = solveLDL3(lhs, rhs);

    // Apply position correction: x -= delta
    body.position.x -= delta[0];
    body.position.y -= delta[1];
    body.angle -= delta[2];
  }

  /**
   * Dual update for a constraint row.
   * Updates lambda (Lagrange multiplier) and ramps penalty.
   * From avbd-demo2d: solver.cpp lines ~210-240
   */
  private dualUpdate(row: ConstraintRow, rowIndex: number, dt: number): void {
    const bodies = this.bodyStore.bodies;

    // Re-evaluate constraint value with current positions
    let cEval = row.c0;
    if (row.bodyA >= 0) {
      const bA = bodies[row.bodyA];
      cEval += row.jacobianA[0] * (bA.position.x - bA.prevPosition.x)
            + row.jacobianA[1] * (bA.position.y - bA.prevPosition.y)
            + row.jacobianA[2] * (bA.angle - bA.prevAngle);
    }
    if (row.bodyB >= 0) {
      const bB = bodies[row.bodyB];
      cEval += row.jacobianB[0] * (bB.position.x - bB.prevPosition.x)
            + row.jacobianB[1] * (bB.position.y - bB.prevPosition.y)
            + row.jacobianB[2] * (bB.angle - bB.prevAngle);
    }

    // For soft constraints (finite stiffness), zero lambda before accumulation
    // Reference: solver.cpp "lambda = isinf(stiffness[i]) ? force->lambda[i] : 0.0f"
    const prevLambda = isFinite(row.stiffness) ? 0 : row.lambda;

    // Update lambda: lambda = clamp(penalty * C + lambda, fmin, fmax)
    row.lambda = row.penalty * cEval + prevLambda;
    row.lambda = Math.max(row.fmin, Math.min(row.fmax, row.lambda));

    // Fracture check
    if (row.fractureThreshold > 0 && Math.abs(row.lambda) > row.fractureThreshold) {
      row.broken = true;
      row.active = false;
      return;
    }

    // Conditional penalty ramping: only ramp when constraint is interior (not at bounds)
    // Reference: "if (lambda > fmin && lambda < fmax) penalty += beta * |C|"
    if (row.lambda > row.fmin && row.lambda < row.fmax) {
      row.penalty += this.config.beta * Math.abs(cEval);
    }
    row.penalty = Math.max(this.config.penaltyMin, Math.min(this.config.penaltyMax, row.penalty));
    if (row.penalty > row.stiffness) {
      row.penalty = row.stiffness;
    }

    // Update friction bounds for contact pairs (normal row updates friction row)
    if (row.type === ForceType.Contact && rowIndex % 2 === 0) {
      // This is a normal row; update the next row (friction row)
      if (rowIndex + 1 < this.constraintStore.rows.length) {
        const frictionRow = this.constraintStore.rows[rowIndex + 1];
        if (frictionRow.active && frictionRow.type === ForceType.Contact) {
          const bodyA = bodies[row.bodyA];
          const bodyB = bodies[row.bodyB];
          updateFrictionBounds(row, frictionRow, bodyA, bodyB);
        }
      }
    }
  }
}
