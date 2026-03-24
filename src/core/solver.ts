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

import type { Vec2, SolverConfig, ContactManifold2D, StepTimings } from './types.js';
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

/**
 * Spatial hash grid broadphase: returns candidate pairs (i, j) that pass AABB overlap.
 * O(n) average-case instead of O(n²).
 */
function broadphase2D(
  bodyStore: BodyStore2D,
  ignorePairs: Set<string>,
): [number, number][] {
  const bodies = bodyStore.bodies;
  const n = bodies.length;
  if (n === 0) return [];

  // Compute all AABBs once
  const aabbs = new Array(n);
  for (let i = 0; i < n; i++) {
    aabbs[i] = bodyStore.getAABB(bodies[i]);
  }

  // Cell size: ~2x average body extent
  let totalExtent = 0;
  let dynamicCount = 0;
  for (let i = 0; i < n; i++) {
    if (bodies[i].type === RigidBodyType.Dynamic) {
      totalExtent += (aabbs[i].maxX - aabbs[i].minX) + (aabbs[i].maxY - aabbs[i].minY);
      dynamicCount++;
    }
  }
  const cellSize = Math.max(dynamicCount > 0 ? totalExtent / (dynamicCount * 2) * 2 : 1, 0.5);
  const invCell = 1 / cellSize;

  // Insert bodies into spatial hash
  const grid = new Map<number, number[]>();

  for (let i = 0; i < n; i++) {
    const aabb = aabbs[i];
    const minCX = Math.floor(aabb.minX * invCell);
    const minCY = Math.floor(aabb.minY * invCell);
    const maxCX = Math.floor(aabb.maxX * invCell);
    const maxCY = Math.floor(aabb.maxY * invCell);

    for (let cx = minCX; cx <= maxCX; cx++) {
      for (let cy = minCY; cy <= maxCY; cy++) {
        const key = (cx + 0x8000) * 0x10000 + (cy + 0x8000);
        let cell = grid.get(key);
        if (!cell) { cell = []; grid.set(key, cell); }
        cell.push(i);
      }
    }
  }

  // Collect candidate pairs
  const pairs: [number, number][] = [];
  const tested = new Set<number>();

  for (const cell of grid.values()) {
    for (let ci = 0; ci < cell.length; ci++) {
      const i = cell[ci];
      for (let cj = ci + 1; cj < cell.length; cj++) {
        const j = cell[cj];
        const pairKey = i < j ? i * n + j : j * n + i;
        if (tested.has(pairKey)) continue;
        tested.add(pairKey);

        const a = bodies[i], b = bodies[j];
        if (a.type !== RigidBodyType.Dynamic && b.type !== RigidBodyType.Dynamic) continue;

        const strKey = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (ignorePairs.has(strKey)) continue;

        if (!aabb2DOverlap(aabbs[i], aabbs[j])) continue;

        pairs.push([i, j]);
      }
    }
  }

  return pairs;
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

  /** Last step's performance breakdown */
  lastTimings: StepTimings | null = null;

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

    const t0 = performance.now();

    // ─── 1. Broadphase & Narrowphase ──────────────────────────────
    constraintStore.clearContacts();

    const tBP = performance.now();
    const candidatePairs = broadphase2D(bodyStore, this.ignorePairs);
    const tNP = performance.now();

    // Narrowphase: SAT collision detection on candidate pairs
    for (const [i, j] of candidatePairs) {
      const manifold = collide2D(bodies[i], bodies[j]);
      if (manifold) {
        const rows = createContactConstraintRows(
          manifold, bodies[i], bodies[j],
          config.penaltyMin, Infinity, config.dt,
        );
        constraintStore.addRows(rows);
      }
    }
    constraintStore.warmstartContacts();
    const tNPEnd = performance.now();

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

    const tWS = performance.now();

    // ─── 3. Initialize Bodies ─────────────────────────────────────
    const MAX_ANGULAR_VELOCITY = 50; // Reference clamps to [-50, 50]
    const gravMag = vec2Length(gravity);

    for (const body of bodies) {
      if (body.type !== RigidBodyType.Dynamic) continue;

      // Clamp velocities at step start to prevent explosive inertial predictions
      const MAX_LINEAR_VELOCITY_INIT = 100;
      const vMagInit = Math.sqrt(body.velocity.x * body.velocity.x + body.velocity.y * body.velocity.y);
      if (vMagInit > MAX_LINEAR_VELOCITY_INIT) {
        const scale = MAX_LINEAR_VELOCITY_INIT / vMagInit;
        body.velocity.x *= scale;
        body.velocity.y *= scale;
      }
      body.angularVelocity = Math.max(-MAX_ANGULAR_VELOCITY,
        Math.min(MAX_ANGULAR_VELOCITY, body.angularVelocity));

      // Save initial position
      body.prevPosition = { ...body.position };
      body.prevAngle = body.angle;

      // Adaptive gravity weighting: only reduce gravity for slow-moving (supported)
      // bodies. Fast-moving bodies need full gravity to avoid artificial bounce.
      let gravWeight = 1;
      if (gravMag > 0) {
        const speed = Math.sqrt(body.velocity.x * body.velocity.x + body.velocity.y * body.velocity.y);
        if (speed < 0.5) {
          const dvx = body.velocity.x - body.prevVelocity.x;
          const dvy = body.velocity.y - body.prevVelocity.y;
          const dvMag = Math.sqrt(dvx * dvx + dvy * dvy);
          if (dvMag > 0.01) {
            const gravDir = { x: gravity.x / gravMag, y: gravity.y / gravMag };
            const accelInGravDir = (dvx * gravDir.x + dvy * gravDir.y) / dt;
            gravWeight = Math.max(0, Math.min(1, accelInGravDir / gravMag));
          }
        }
      }

      // Apply velocity damping.
      // A small implicit angular damping (0.05) provides numerical dissipation
      // that prevents contact-induced angular velocity feedback loops in stacking
      // scenarios (pyramids, multi-body piles). This doesn't affect the physics
      // meaningfully but prevents solver artifacts from amplifying.
      const IMPLICIT_ANGULAR_DAMPING = 0.05;
      let vx = body.velocity.x;
      let vy = body.velocity.y;
      let omega = body.angularVelocity;

      if (body.linearDamping > 0) {
        const dampFactor = 1 / (1 + body.linearDamping * dt);
        vx *= dampFactor;
        vy *= dampFactor;
      }
      {
        const totalAngDamp = body.angularDamping + IMPLICIT_ANGULAR_DAMPING;
        const dampFactor = 1 / (1 + totalAngDamp * dt);
        omega *= dampFactor;
      }

      // Inertial target uses FULL gravity (the optimization objective target).
      // Reference: solver.cpp — body->inertialLin = pos + vel*dt + gravity*dt²
      body.inertialPosition = {
        x: body.prevPosition.x + vx * dt + gravity.x * body.gravityScale * dt * dt,
        y: body.prevPosition.y + vy * dt + gravity.y * body.gravityScale * dt * dt,
      };
      body.inertialAngle = body.prevAngle + omega * dt;

      // Move body to predicted position (initial guess for solver).
      // Uses adaptive gravity weight so resting bodies don't over-predict penetration.
      // Reference: solver.cpp — body->positionLin = pos + vel*dt + gravity*accelWeight*dt²
      body.position = {
        x: body.prevPosition.x + vx * dt + gravity.x * body.gravityScale * gravWeight * dt * dt,
        y: body.prevPosition.y + vy * dt + gravity.y * body.gravityScale * gravWeight * dt * dt,
      };
      body.angle = body.prevAngle + omega * dt;

      // Save velocity for next frame's adaptive gravity
      body.prevVelocity = { ...body.velocity };
    }

    const tBI = performance.now();

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

          // Clamp recovered linear velocity
          const MAX_LINEAR_VELOCITY = 100;
          const vMag = Math.sqrt(body.velocity.x * body.velocity.x + body.velocity.y * body.velocity.y);
          if (vMag > MAX_LINEAR_VELOCITY) {
            const scale = MAX_LINEAR_VELOCITY / vMag;
            body.velocity.x *= scale;
            body.velocity.y *= scale;
          }

          body.angularVelocity = (body.angle - body.prevAngle) / dt;

          // Clamp recovered angular velocity to prevent explosive inertial predictions
          body.angularVelocity = Math.max(-MAX_ANGULAR_VELOCITY,
            Math.min(MAX_ANGULAR_VELOCITY, body.angularVelocity));
        }
      }
    }

    const tEnd = performance.now();
    this.lastTimings = {
      total: tEnd - t0,
      broadphase: tNP - tBP,
      narrowphase: tNPEnd - tNP,
      warmstart: tWS - tNPEnd,
      bodyInit: tBI - tWS,
      solverIters: tEnd - tBI,
      velocityRecover: 0,
      numBodies: bodies.length,
      numConstraints: constraintStore.rows.length,
    };
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
      // The (1-alpha) factor prevents over-correction of pre-existing penetrations.
      // Reference: manifold.cpp — C = C0 * (1 - alpha) + J*dq
      let cEval = row.c0 * (1 - this.config.alpha);
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
    // C = C0*(1-alpha) + J*dp (same linearization as primal)
    let cEval = row.c0 * (1 - this.config.alpha);
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

    // Conditional penalty ramping: only ramp when constraint is interior (not at bounds).
    // For normal contacts: ramp when lambda < 0 (contact active, not at fmax=0 bound).
    // For friction: ramp when NOT sliding (interior of friction cone).
    // Reference: manifold.cpp — penalty += beta * |C| when active/sticking
    if (row.lambda > row.fmin && row.lambda < row.fmax) {
      row.penalty += this.config.beta * Math.abs(cEval);
    }
    row.penalty = Math.max(this.config.penaltyMin, Math.min(this.config.penaltyMax, row.penalty));
    if (row.penalty > row.stiffness) {
      row.penalty = row.stiffness;
    }

    // Update friction bounds for contact pairs (normal row updates friction row).
    // Normal rows are unilateral: fmin=-Infinity, fmax=0. Friction rows have finite bounds.
    // We identify normal rows by their infinite fmin (never modified by friction coupling).
    if (row.type === ForceType.Contact && !isFinite(row.fmin)) {
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
