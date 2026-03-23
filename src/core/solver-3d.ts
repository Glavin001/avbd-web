/**
 * AVBD 3D Solver — 6-DOF rigid body simulation.
 * Position: (x, y, z), Orientation: quaternion (w, x, y, z)
 * State vector for primal update: [dx, dy, dz, dwx, dwy, dwz] (6-DOF)
 *
 * The 3D solver follows the same AVBD algorithm as the 2D solver:
 * 1. Broadphase → 2. Warmstart → 3. Primal update (6x6 LDL) → 4. Dual update
 */

import type { Vec3, Quat, SolverConfig } from './types.js';
import { RigidBodyType, DEFAULT_SOLVER_CONFIG_3D } from './types.js';
import { ForceType } from './types.js';
import type { Body3D } from './rigid-body-3d.js';
import { BodyStore3D } from './rigid-body-3d.js';
import { collide3D, getAABB3D, aabb3DOverlap, type ContactManifold3D } from '../3d/collision-gjk.js';
import {
  vec3, vec3Add, vec3Sub, vec3Scale, vec3Cross, vec3Dot, vec3Length,
  quatIdentity, quatMul, quatNormalize, quatRotateVec3, quatFromAxisAngle,
} from './math.js';
import { COLLISION_MARGIN } from './types.js';

// ─── 3D Constraint Row ─────────────────────────────────────────────────────

export interface ConstraintRow3D {
  bodyA: number;
  bodyB: number;
  type: ForceType;
  /** Jacobian for body A: 6-DOF [dx, dy, dz, dwx, dwy, dwz] */
  jacobianA: number[];
  /** Jacobian for body B */
  jacobianB: number[];
  /** Diagonal of the Hessian for geometric stiffness (body A, 6-DOF) */
  hessianDiagA: number[];
  /** Diagonal of the Hessian for geometric stiffness (body B, 6-DOF) */
  hessianDiagB: number[];
  c: number;
  c0: number;
  lambda: number;
  penalty: number;
  stiffness: number;
  fmin: number;
  fmax: number;
  active: boolean;
  broken: boolean;
}

function createDefaultRow3D(): ConstraintRow3D {
  return {
    bodyA: -1, bodyB: -1,
    type: ForceType.Contact,
    jacobianA: [0, 0, 0, 0, 0, 0],
    jacobianB: [0, 0, 0, 0, 0, 0],
    hessianDiagA: [0, 0, 0, 0, 0, 0],
    hessianDiagB: [0, 0, 0, 0, 0, 0],
    c: 0, c0: 0,
    lambda: 0, penalty: 100,
    stiffness: Infinity,
    fmin: -Infinity, fmax: Infinity,
    active: true, broken: false,
  };
}

// ─── Contact constraint creation (3D) ───────────────────────────────────────

function createContactRows3D(
  manifold: ContactManifold3D,
  bodyA: Body3D,
  bodyB: Body3D,
  penaltyMin: number,
): ConstraintRow3D[] {
  const rows: ConstraintRow3D[] = [];
  const mu = Math.sqrt(bodyA.friction * bodyB.friction);

  for (const contact of manifold.contacts) {
    const n = manifold.normal;
    const rA = vec3Sub(contact.position, bodyA.position);
    const rB = vec3Sub(contact.position, bodyB.position);

    // Normal constraint
    const nRow = createDefaultRow3D();
    nRow.bodyA = manifold.bodyA;
    nRow.bodyB = manifold.bodyB;
    nRow.type = ForceType.Contact;

    // J_A = [n, rA × n] for body A
    const torqueA = vec3Cross(rA, n);
    nRow.jacobianA = [n.x, n.y, n.z, torqueA.x, torqueA.y, torqueA.z];

    // J_B = [-n, -(rB × n)] for body B (opposing)
    const torqueB = vec3Cross(rB, n);
    nRow.jacobianB = [-n.x, -n.y, -n.z, -torqueB.x, -torqueB.y, -torqueB.z];

    // Geometric stiffness (Hessian diagonal) for angular DOFs:
    // H[3+i] = -(r·n) + r[i]*n[i], the second derivative of constraint w.r.t. rotation
    const rAdotN = vec3Dot(rA, n);
    const rBdotN = vec3Dot(rB, n);
    nRow.hessianDiagA = [0, 0, 0,
      -(rAdotN - rA.x * n.x),
      -(rAdotN - rA.y * n.y),
      -(rAdotN - rA.z * n.z)];
    nRow.hessianDiagB = [0, 0, 0,
      -(rBdotN - rB.x * n.x),
      -(rBdotN - rB.y * n.y),
      -(rBdotN - rB.z * n.z)];

    nRow.c = -contact.depth + COLLISION_MARGIN;
    nRow.c0 = nRow.c;
    nRow.fmin = -Infinity;
    nRow.fmax = 0;
    nRow.penalty = penaltyMin;

    rows.push(nRow);

    // Friction constraints (two tangent directions)
    const t1 = computeTangent(n);
    const t2 = vec3Cross(n, t1);

    for (const t of [t1, t2]) {
      const fRow = createDefaultRow3D();
      fRow.bodyA = manifold.bodyA;
      fRow.bodyB = manifold.bodyB;
      fRow.type = ForceType.Contact;

      const tA = vec3Cross(rA, t);
      fRow.jacobianA = [t.x, t.y, t.z, tA.x, tA.y, tA.z];

      const tB = vec3Cross(rB, t);
      fRow.jacobianB = [-t.x, -t.y, -t.z, -tB.x, -tB.y, -tB.z];

      // Geometric stiffness for friction tangent direction
      const rAdotT = vec3Dot(rA, t);
      const rBdotT = vec3Dot(rB, t);
      fRow.hessianDiagA = [0, 0, 0,
        -(rAdotT - rA.x * t.x),
        -(rAdotT - rA.y * t.y),
        -(rAdotT - rA.z * t.z)];
      fRow.hessianDiagB = [0, 0, 0,
        -(rBdotT - rB.x * t.x),
        -(rBdotT - rB.y * t.y),
        -(rBdotT - rB.z * t.z)];

      fRow.c = 0;
      fRow.c0 = 0;
      fRow.fmin = -mu * penaltyMin * contact.depth;
      fRow.fmax = mu * penaltyMin * contact.depth;
      fRow.penalty = penaltyMin;

      rows.push(fRow);
    }
  }
  return rows;
}

function computeTangent(n: Vec3): Vec3 {
  const up = Math.abs(n.y) < 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0);
  const t = vec3Cross(n, up);
  const len = vec3Length(t);
  return len > 1e-10 ? vec3Scale(t, 1 / len) : vec3(1, 0, 0);
}

// ─── 6x6 LDL^T Solver ──────────────────────────────────────────────────────

function solveLDL6(A: number[], b: number[]): number[] {
  const n = 6;
  const L: number[] = new Array(n * n).fill(0);
  const D: number[] = new Array(n).fill(0);

  for (let j = 0; j < n; j++) {
    let sumD = A[j * n + j];
    for (let k = 0; k < j; k++) {
      sumD -= L[k * n + j] * L[k * n + j] * D[k];
    }
    D[j] = sumD;

    for (let i = j + 1; i < n; i++) {
      let sumL = A[j * n + i];
      for (let k = 0; k < j; k++) {
        sumL -= L[k * n + i] * L[k * n + j] * D[k];
      }
      L[j * n + i] = sumL / D[j];
    }
  }

  // Forward substitution
  const y: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[k * n + i] * y[k];
    y[i] = s;
  }

  // Diagonal
  const z: number[] = new Array(n);
  for (let i = 0; i < n; i++) z[i] = y[i] / D[i];

  // Back substitution
  const x: number[] = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < n; k++) s -= L[i * n + k] * x[k];
    x[i] = s;
  }

  return x;
}

// ─── 3D AVBD Solver ─────────────────────────────────────────────────────────

export class AVBDSolver3D {
  config: SolverConfig;
  bodyStore: BodyStore3D;
  constraintRows: ConstraintRow3D[] = [];
  ignorePairs: Set<string> = new Set();
  jointRows: ConstraintRow3D[] = [];

  constructor(config: Partial<SolverConfig> = {}) {
    this.config = { ...DEFAULT_SOLVER_CONFIG_3D, ...config };
    this.bodyStore = new BodyStore3D();
  }

  step(): void {
    const { config, bodyStore } = this;
    const dt = config.dt;
    const gravity = config.gravity as Vec3;
    const bodies = bodyStore.bodies;
    if (bodies.length === 0) return;

    // 1. Clear contact rows, keep joint rows
    this.constraintRows = [...this.jointRows];

    // 2. Broadphase + narrowphase
    for (let i = 0; i < bodies.length; i++) {
      const a = bodies[i];
      const aabbA = getAABB3D(a);
      for (let j = i + 1; j < bodies.length; j++) {
        const b = bodies[j];
        if (a.type === RigidBodyType.Fixed && b.type === RigidBodyType.Fixed) continue;
        const key = `${i}-${j}`;
        if (this.ignorePairs.has(key)) continue;
        if (!aabb3DOverlap(aabbA, getAABB3D(b))) continue;
        const manifold = collide3D(a, b);
        if (manifold) {
          const rows = createContactRows3D(manifold, a, b, config.penaltyMin);
          this.constraintRows.push(...rows);
        }
      }
    }

    // 3. Warmstart
    for (const row of this.constraintRows) {
      if (!row.active) continue;
      row.penalty *= config.gamma;
      row.penalty = Math.max(config.penaltyMin, Math.min(config.penaltyMax, row.penalty));
      if (row.penalty > row.stiffness) row.penalty = row.stiffness;
    }

    // 4. Initialize bodies
    const MAX_ANG_VEL = 50;
    const MAX_LIN_VEL = 100;
    const gravMag = vec3Length(gravity);

    for (const body of bodies) {
      if (body.type === RigidBodyType.Fixed) continue;

      // Clamp velocities at step start to prevent explosive inertial predictions
      const linSpeed = vec3Length(body.velocity);
      if (linSpeed > MAX_LIN_VEL) {
        body.velocity = vec3Scale(body.velocity, MAX_LIN_VEL / linSpeed);
      }
      const angSpeed = vec3Length(body.angularVelocity);
      if (angSpeed > MAX_ANG_VEL) {
        body.angularVelocity = vec3Scale(body.angularVelocity, MAX_ANG_VEL / angSpeed);
      }

      // Implicit angular damping: prevents contact-induced angular velocity
      // feedback loops in stacking scenarios (pyramids, multi-body piles).
      const IMPLICIT_ANGULAR_DAMPING = 0.05;
      const angDampFactor = 1 / (1 + IMPLICIT_ANGULAR_DAMPING * dt);
      body.angularVelocity = vec3Scale(body.angularVelocity, angDampFactor);

      body.prevPosition = { ...body.position };
      body.prevRotation = { ...body.rotation };

      // Adaptive gravity weighting (from 2D reference solver):
      // Bodies under support (nearly stationary) get less gravity in the inertial
      // estimate, preventing artificial penetrations that cause explosive corrections.
      // Only apply to slow-moving bodies to avoid creating artificial bounce on impact.
      let gravWeight = 1;
      if (gravMag > 0 && body.prevVelocity) {
        const speed = vec3Length(body.velocity);
        // Only reduce gravity for slow-moving bodies (supported/resting).
        // Fast-moving bodies need full gravity to fall naturally.
        if (speed < 0.5) {
          const dvx = body.velocity.x - body.prevVelocity.x;
          const dvy = body.velocity.y - body.prevVelocity.y;
          const dvz = body.velocity.z - body.prevVelocity.z;
          const dvMag = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
          if (dvMag > 0.01) {
            const gravDir = vec3Scale(gravity, 1 / gravMag);
            const accelInGravDir = (dvx * gravDir.x + dvy * gravDir.y + dvz * gravDir.z) / dt;
            gravWeight = Math.max(0, Math.min(1, accelInGravDir / gravMag));
          }
        }
      }
      body.prevVelocity = { ...body.velocity };

      body.inertialPosition = {
        x: body.position.x + body.velocity.x * dt + gravity.x * body.gravityScale * gravWeight * dt * dt,
        y: body.position.y + body.velocity.y * dt + gravity.y * body.gravityScale * gravWeight * dt * dt,
        z: body.position.z + body.velocity.z * dt + gravity.z * body.gravityScale * gravWeight * dt * dt,
      };
      // Inertial rotation: integrate angular velocity
      const wLen = vec3Length(body.angularVelocity);
      if (wLen > 1e-10) {
        const axis = vec3Scale(body.angularVelocity, 1 / wLen);
        const dq = quatFromAxisAngle(axis, wLen * dt);
        body.inertialRotation = quatNormalize(quatMul(dq, body.rotation));
      } else {
        body.inertialRotation = { ...body.rotation };
      }
    }

    // 5. Solver iterations
    const totalIters = config.postStabilize ? config.iterations + 1 : config.iterations;
    for (let iter = 0; iter < totalIters; iter++) {
      const isStab = config.postStabilize && iter === totalIters - 1;

      // Primal update
      for (const body of bodies) {
        if (body.type === RigidBodyType.Fixed) continue;
        this.primalUpdate3D(body, dt);
      }

      // Dual update (skip on stabilization)
      if (!isStab) {
        for (const row of this.constraintRows) {
          if (!row.active || row.broken) continue;
          this.dualUpdate3D(row, dt);
        }

        // Friction coupling: update friction bounds from normal lambdas.
        // 3D contacts come in triplets: [normal, friction1, friction2].
        // Walk rows finding normal rows (fmax <= 0) and update the next 2 friction rows.
        for (let i = 0; i + 2 < this.constraintRows.length; i++) {
          const normalRow = this.constraintRows[i];
          // Identify normal rows: Contact type with infinite fmin (unilateral constraint)
          if (normalRow.type !== ForceType.Contact || !normalRow.active || isFinite(normalRow.fmin)) continue;
          const fric1 = this.constraintRows[i + 1];
          const fric2 = this.constraintRows[i + 2];
          if (!fric1.active || fric1.type !== ForceType.Contact) continue;
          if (!fric2.active || fric2.type !== ForceType.Contact) continue;
          const bA = this.bodyStore.bodies[normalRow.bodyA];
          const bB = this.bodyStore.bodies[normalRow.bodyB];
          const mu = Math.sqrt(bA.friction * bB.friction);
          const normalForce = Math.abs(normalRow.lambda);
          fric1.fmin = -mu * normalForce;
          fric1.fmax = mu * normalForce;
          fric2.fmin = -mu * normalForce;
          fric2.fmax = mu * normalForce;
          i += 2; // Skip past the friction rows
        }
      }

      // Velocity recovery at iteration N-1
      if (iter === config.iterations - 1) {
        for (const body of bodies) {
          if (body.type === RigidBodyType.Fixed) continue;
          body.velocity = vec3Scale(vec3Sub(body.position, body.prevPosition), 1 / dt);

          // Clamp recovered linear velocity
          const vLen = vec3Length(body.velocity);
          if (vLen > MAX_LIN_VEL) {
            body.velocity = vec3Scale(body.velocity, MAX_LIN_VEL / vLen);
          }

          // Angular velocity from quaternion difference
          const dq = quatMul(body.rotation, {
            w: body.prevRotation.w,
            x: -body.prevRotation.x,
            y: -body.prevRotation.y,
            z: -body.prevRotation.z,
          });
          body.angularVelocity = vec3Scale(vec3(dq.x, dq.y, dq.z), 2 / dt);

          // Clamp recovered angular velocity
          const wLen = vec3Length(body.angularVelocity);
          if (wLen > MAX_ANG_VEL) {
            body.angularVelocity = vec3Scale(body.angularVelocity, MAX_ANG_VEL / wLen);
          }
        }

        // Velocity-level restitution: correct recovered velocity along contact normals.
        // Without this, the PBD position correction creates artificial bounce —
        // the constraint pushes the body out of penetration, and velocity recovery
        // captures this correction as an upward velocity.
        for (const row of this.constraintRows) {
          if (!row.active || row.broken) continue;
          // Only apply to normal contact rows (unilateral, fmin=-Inf, fmax=0)
          if (row.type !== ForceType.Contact || isFinite(row.fmin)) continue;
          if (row.lambda >= -1e-6) continue; // Skip inactive contacts

          const bAidx = row.bodyA;
          const bBidx = row.bodyB;
          // Extract contact normal from Jacobian (first 3 components for body A)
          const n = vec3(row.jacobianA[0], row.jacobianA[1], row.jacobianA[2]);

          // Apply restitution to body A
          if (bAidx >= 0 && bodies[bAidx].type === RigidBodyType.Dynamic) {
            const bA = bodies[bAidx];
            const vnA = vec3Dot(bA.velocity, n);
            if (vnA > 0) { // Separating velocity along normal (bounce)
              const restitution = Math.max(bA.restitution, bodies[bBidx]?.restitution ?? 0);
              // Remove the artificial bounce, add back restitution amount
              const correction = vnA * (1 - restitution);
              bA.velocity = vec3Sub(bA.velocity, vec3Scale(n, correction));
            }
          }

          // Apply restitution to body B (opposite normal)
          if (bBidx >= 0 && bodies[bBidx].type === RigidBodyType.Dynamic) {
            const bB = bodies[bBidx];
            const vnB = vec3Dot(bB.velocity, n);
            if (vnB < 0) { // Separating in opposite direction
              const restitution = Math.max(bodies[bAidx]?.restitution ?? 0, bB.restitution);
              const correction = -vnB * (1 - restitution);
              bB.velocity = vec3Add(bB.velocity, vec3Scale(n, correction));
            }
          }
        }
      }
    }
  }

  private primalUpdate3D(body: Body3D, dt: number): void {
    const dt2 = dt * dt;
    const n = 6;

    // LHS = M/dt^2 (6x6 diagonal mass matrix)
    const lhs: number[] = new Array(n * n).fill(0);
    lhs[0 * n + 0] = body.mass / dt2;
    lhs[1 * n + 1] = body.mass / dt2;
    lhs[2 * n + 2] = body.mass / dt2;
    lhs[3 * n + 3] = body.inertia.x / dt2;
    lhs[4 * n + 4] = body.inertia.y / dt2;
    lhs[5 * n + 5] = body.inertia.z / dt2;

    // Position error for RHS
    const dp = vec3Sub(body.position, body.inertialPosition);
    // Rotation error: small angle approximation
    const dq = quatMul(body.rotation, {
      w: body.inertialRotation.w,
      x: -body.inertialRotation.x,
      y: -body.inertialRotation.y,
      z: -body.inertialRotation.z,
    });
    const dtheta: Vec3 = vec3Scale(vec3(dq.x, dq.y, dq.z), 2);

    const rhs: number[] = [
      body.mass / dt2 * dp.x,
      body.mass / dt2 * dp.y,
      body.mass / dt2 * dp.z,
      body.inertia.x / dt2 * dtheta.x,
      body.inertia.y / dt2 * dtheta.y,
      body.inertia.z / dt2 * dtheta.z,
    ];

    // Accumulate constraints
    for (const row of this.constraintRows) {
      if (!row.active || row.broken) continue;

      let J: number[] | null = null;
      if (row.bodyA === body.index) J = row.jacobianA;
      else if (row.bodyB === body.index) J = row.jacobianB;
      else continue;

      // Evaluate linearized constraint
      let cEval = row.c0;
      if (row.bodyA >= 0) {
        const bA = this.bodyStore.bodies[row.bodyA];
        const dpA = vec3Sub(bA.position, bA.prevPosition);
        const dqA = quatMul(bA.rotation, {
          w: bA.prevRotation.w, x: -bA.prevRotation.x,
          y: -bA.prevRotation.y, z: -bA.prevRotation.z,
        });
        const dthetaA = [dpA.x, dpA.y, dpA.z, 2 * dqA.x, 2 * dqA.y, 2 * dqA.z];
        for (let k = 0; k < 6; k++) cEval += row.jacobianA[k] * dthetaA[k];
      }
      if (row.bodyB >= 0) {
        const bB = this.bodyStore.bodies[row.bodyB];
        const dpB = vec3Sub(bB.position, bB.prevPosition);
        const dqB = quatMul(bB.rotation, {
          w: bB.prevRotation.w, x: -bB.prevRotation.x,
          y: -bB.prevRotation.y, z: -bB.prevRotation.z,
        });
        const dthetaB = [dpB.x, dpB.y, dpB.z, 2 * dqB.x, 2 * dqB.y, 2 * dqB.z];
        for (let k = 0; k < 6; k++) cEval += row.jacobianB[k] * dthetaB[k];
      }

      let f = row.penalty * cEval + row.lambda;
      f = Math.max(row.fmin, Math.min(row.fmax, f));

      // RHS += J * f
      for (let k = 0; k < n; k++) rhs[k] += J[k] * f;

      // LHS += J * J^T * penalty
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          lhs[j * n + i] += J[i] * J[j] * row.penalty;
        }
      }

      // Geometric stiffness (diagonal lumping)
      const hDiag = row.bodyA === body.index ? row.hessianDiagA : row.hessianDiagB;
      const absF = Math.abs(f);
      for (let i = 0; i < n; i++) {
        lhs[i * n + i] += Math.abs(hDiag[i]) * absF;
      }
    }

    // Check diagonal validity
    for (let i = 0; i < n; i++) {
      if (lhs[i * n + i] <= 0) return;
    }

    const delta = solveLDL6(lhs, rhs);

    // Apply position correction
    body.position.x -= delta[0];
    body.position.y -= delta[1];
    body.position.z -= delta[2];

    // Apply rotation correction
    const wDelta = vec3(-delta[3], -delta[4], -delta[5]);
    const wLen = vec3Length(wDelta);
    if (wLen > 1e-10) {
      const axis = vec3Scale(wDelta, 1 / wLen);
      const dqCorr = quatFromAxisAngle(axis, wLen);
      body.rotation = quatNormalize(quatMul(dqCorr, body.rotation));
    }
  }

  private dualUpdate3D(row: ConstraintRow3D, dt: number): void {
    let cEval = row.c0;
    if (row.bodyA >= 0) {
      const bA = this.bodyStore.bodies[row.bodyA];
      const dpA = vec3Sub(bA.position, bA.prevPosition);
      const dqA = quatMul(bA.rotation, {
        w: bA.prevRotation.w, x: -bA.prevRotation.x,
        y: -bA.prevRotation.y, z: -bA.prevRotation.z,
      });
      const dthetaA = [dpA.x, dpA.y, dpA.z, 2 * dqA.x, 2 * dqA.y, 2 * dqA.z];
      for (let k = 0; k < 6; k++) cEval += row.jacobianA[k] * dthetaA[k];
    }
    if (row.bodyB >= 0) {
      const bB = this.bodyStore.bodies[row.bodyB];
      const dpB = vec3Sub(bB.position, bB.prevPosition);
      const dqB = quatMul(bB.rotation, {
        w: bB.prevRotation.w, x: -bB.prevRotation.x,
        y: -bB.prevRotation.y, z: -bB.prevRotation.z,
      });
      const dthetaB = [dpB.x, dpB.y, dpB.z, 2 * dqB.x, 2 * dqB.y, 2 * dqB.z];
      for (let k = 0; k < 6; k++) cEval += row.jacobianB[k] * dthetaB[k];
    }

    row.lambda = Math.max(row.fmin, Math.min(row.fmax, row.penalty * cEval + row.lambda));
    // Conditional penalty ramp: only when lambda is interior (not at bounds).
    // Skip ramping for friction rows (Contact type with finite fmin) — friction penalty
    // should stay low to avoid stiff angular springs that cause spinning instability.
    const isFrictionRow = row.type === ForceType.Contact && isFinite(row.fmin);
    if (!isFrictionRow && row.lambda > row.fmin && row.lambda < row.fmax) {
      // Cap the per-iteration penalty increase to prevent exponential growth
      // that causes explosive forces in many-body scenes (pyramids, stacks).
      // Allow at most 50% growth per iteration (1.5^10 ≈ 57x max per step).
      const increment = this.config.beta * Math.abs(cEval);
      const maxIncrement = row.penalty * 0.5;
      row.penalty += Math.min(increment, maxIncrement);
    }
    row.penalty = Math.max(this.config.penaltyMin, Math.min(this.config.penaltyMax, row.penalty));
    if (row.penalty > row.stiffness) row.penalty = row.stiffness;
  }
}
