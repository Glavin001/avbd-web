/**
 * AVBD 3D Solver — 6-DOF rigid body simulation.
 * Position: (x, y, z), Orientation: quaternion (w, x, y, z)
 * State vector for primal update: [dx, dy, dz, dwx, dwy, dwz] (6-DOF)
 *
 * The 3D solver follows the same AVBD algorithm as the 2D solver:
 * 1. Broadphase → 2. Warmstart → 3. Primal update (6x6 LDL) → 4. Dual update
 */

import type { Vec3, Quat, SolverConfig, StepTimings } from './types.js';
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

function solveLDL6(A: ArrayLike<number>, b: ArrayLike<number>): number[] {
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
  lastTimings: StepTimings | null = null;

  // Pooled arrays for primalUpdate3D (avoid per-body per-iteration allocation)
  private _lhs = new Float64Array(36);
  private _rhs = new Float64Array(6);

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

    const t0 = performance.now();

    // 1. Clear contact rows, keep joint rows
    this.constraintRows = [...this.jointRows];

    // 2. Broadphase (spatial hash) — find candidate pairs
    const candidatePairs3D: [number, number][] = [];
    {
      const n = bodies.length;
      const aabbs = new Array(n);
      for (let i = 0; i < n; i++) aabbs[i] = getAABB3D(bodies[i]);

      let totalExtent = 0, dynCount = 0;
      for (let i = 0; i < n; i++) {
        if (bodies[i].type !== RigidBodyType.Fixed) {
          const a = aabbs[i];
          totalExtent += (a.max.x - a.min.x) + (a.max.y - a.min.y) + (a.max.z - a.min.z);
          dynCount++;
        }
      }
      const cellSize = Math.max(dynCount > 0 ? (totalExtent / (dynCount * 3)) * 2 : 1, 0.5);
      const invCell = 1 / cellSize;

      const grid = new Map<number, number[]>();
      function hashKey3D(cx: number, cy: number, cz: number): number {
        return ((cx + 0x400) * 0x100000) + ((cy + 0x400) * 0x800) + (cz + 0x400);
      }

      for (let i = 0; i < n; i++) {
        const a = aabbs[i];
        const x0 = Math.floor(a.min.x * invCell), x1 = Math.floor(a.max.x * invCell);
        const y0 = Math.floor(a.min.y * invCell), y1 = Math.floor(a.max.y * invCell);
        const z0 = Math.floor(a.min.z * invCell), z1 = Math.floor(a.max.z * invCell);
        for (let cx = x0; cx <= x1; cx++) {
          for (let cy = y0; cy <= y1; cy++) {
            for (let cz = z0; cz <= z1; cz++) {
              const k = hashKey3D(cx, cy, cz);
              let cell = grid.get(k);
              if (!cell) { cell = []; grid.set(k, cell); }
              cell.push(i);
            }
          }
        }
      }

      const tested = new Set<number>();
      for (const cell of grid.values()) {
        for (let ci = 0; ci < cell.length; ci++) {
          const i = cell[ci];
          for (let cj = ci + 1; cj < cell.length; cj++) {
            const j = cell[cj];
            const pk = i < j ? i * n + j : j * n + i;
            if (tested.has(pk)) continue;
            tested.add(pk);
            if (bodies[i].type === RigidBodyType.Fixed && bodies[j].type === RigidBodyType.Fixed) continue;
            const key = `${i}-${j}`;
            if (this.ignorePairs.has(key)) continue;
            if (!aabb3DOverlap(aabbs[i], aabbs[j])) continue;
            candidatePairs3D.push([i, j]);
          }
        }
      }
    }

    const tBP = performance.now();

    // Narrowphase: GJK collision detection on candidate pairs
    for (const [i, j] of candidatePairs3D) {
      const manifold = collide3D(bodies[i], bodies[j]);
      if (manifold) {
        const rows = createContactRows3D(manifold, bodies[i], bodies[j], config.penaltyMin);
        this.constraintRows.push(...rows);
      }
    }

    const tNP = performance.now();

    // 3. Warmstart
    for (const row of this.constraintRows) {
      if (!row.active) continue;
      row.penalty *= config.gamma;
      row.penalty = Math.max(config.penaltyMin, Math.min(config.penaltyMax, row.penalty));
      if (row.penalty > row.stiffness) row.penalty = row.stiffness;
    }

    const tWS = performance.now();

    // 4. Initialize bodies (in-place mutation to avoid object allocation)
    const MAX_ANG_VEL = 50;
    const MAX_LIN_VEL = 100;
    const gravMag = vec3Length(gravity);
    const gravDirX = gravMag > 0 ? gravity.x / gravMag : 0;
    const gravDirY = gravMag > 0 ? gravity.y / gravMag : 0;
    const gravDirZ = gravMag > 0 ? gravity.z / gravMag : 0;
    const angDampFactor = 1 / (1 + 0.05 * dt);
    const dt2 = dt * dt;

    for (const body of bodies) {
      if (body.type === RigidBodyType.Fixed) continue;

      // Clamp velocities at step start
      const vx = body.velocity.x, vy = body.velocity.y, vz = body.velocity.z;
      const linSpeed = Math.sqrt(vx * vx + vy * vy + vz * vz);
      if (linSpeed > MAX_LIN_VEL) {
        const s = MAX_LIN_VEL / linSpeed;
        body.velocity.x *= s; body.velocity.y *= s; body.velocity.z *= s;
      }
      const awx = body.angularVelocity.x, awy = body.angularVelocity.y, awz = body.angularVelocity.z;
      const angSpeed = Math.sqrt(awx * awx + awy * awy + awz * awz);
      if (angSpeed > MAX_ANG_VEL) {
        const s = MAX_ANG_VEL / angSpeed;
        body.angularVelocity.x *= s; body.angularVelocity.y *= s; body.angularVelocity.z *= s;
      }

      // Implicit angular damping (in-place)
      body.angularVelocity.x *= angDampFactor;
      body.angularVelocity.y *= angDampFactor;
      body.angularVelocity.z *= angDampFactor;

      // Save prev state in-place
      body.prevPosition.x = body.position.x;
      body.prevPosition.y = body.position.y;
      body.prevPosition.z = body.position.z;
      body.prevRotation.w = body.rotation.w;
      body.prevRotation.x = body.rotation.x;
      body.prevRotation.y = body.rotation.y;
      body.prevRotation.z = body.rotation.z;

      // Adaptive gravity weighting
      let gravWeight = 1;
      if (gravMag > 0 && body.prevVelocity) {
        const speed = Math.sqrt(body.velocity.x * body.velocity.x + body.velocity.y * body.velocity.y + body.velocity.z * body.velocity.z);
        if (speed < 0.5) {
          const dvx = body.velocity.x - body.prevVelocity.x;
          const dvy = body.velocity.y - body.prevVelocity.y;
          const dvz = body.velocity.z - body.prevVelocity.z;
          const dvMag = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
          if (dvMag > 0.01) {
            const accelInGravDir = (dvx * gravDirX + dvy * gravDirY + dvz * gravDirZ) / dt;
            gravWeight = Math.max(0, Math.min(1, accelInGravDir / gravMag));
          }
        }
      }
      // Save prev velocity in-place
      body.prevVelocity.x = body.velocity.x;
      body.prevVelocity.y = body.velocity.y;
      body.prevVelocity.z = body.velocity.z;

      // Inertial target uses FULL gravity (the optimization objective target).
      // In-place mutation to avoid object allocation.
      const gsFullDt2 = body.gravityScale * dt2;
      body.inertialPosition.x = body.prevPosition.x + body.velocity.x * dt + gravity.x * gsFullDt2;
      body.inertialPosition.y = body.prevPosition.y + body.velocity.y * dt + gravity.y * gsFullDt2;
      body.inertialPosition.z = body.prevPosition.z + body.velocity.z * dt + gravity.z * gsFullDt2;

      // Inertial rotation: integrate angular velocity (inline to avoid allocations)
      const wax = body.angularVelocity.x, way = body.angularVelocity.y, waz = body.angularVelocity.z;
      const wLen = Math.sqrt(wax * wax + way * way + waz * waz);
      if (wLen > 1e-10) {
        const invWLen = 1 / wLen;
        const halfAngle = wLen * dt * 0.5;
        const s = Math.sin(halfAngle) * invWLen;
        const dqw = Math.cos(halfAngle);
        const dqx = wax * s, dqy = way * s, dqz = waz * s;
        // Quaternion multiply dq * prevRotation (inline)
        const rw = body.prevRotation.w, rx = body.prevRotation.x, ry = body.prevRotation.y, rz = body.prevRotation.z;
        const nw = dqw * rw - dqx * rx - dqy * ry - dqz * rz;
        const nx = dqw * rx + dqx * rw + dqy * rz - dqz * ry;
        const ny = dqw * ry - dqx * rz + dqy * rw + dqz * rx;
        const nz = dqw * rz + dqx * ry - dqy * rx + dqz * rw;
        const qLen = Math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
        const invQLen = 1 / qLen;
        body.inertialRotation.w = nw * invQLen;
        body.inertialRotation.x = nx * invQLen;
        body.inertialRotation.y = ny * invQLen;
        body.inertialRotation.z = nz * invQLen;
        // Also set body rotation (initial guess uses same rotation integration)
        body.rotation.w = nw * invQLen;
        body.rotation.x = nx * invQLen;
        body.rotation.y = ny * invQLen;
        body.rotation.z = nz * invQLen;
      } else {
        body.inertialRotation.w = body.prevRotation.w;
        body.inertialRotation.x = body.prevRotation.x;
        body.inertialRotation.y = body.prevRotation.y;
        body.inertialRotation.z = body.prevRotation.z;
        body.rotation.w = body.prevRotation.w;
        body.rotation.x = body.prevRotation.x;
        body.rotation.y = body.prevRotation.y;
        body.rotation.z = body.prevRotation.z;
      }
      // Move body to predicted position (initial guess, adaptive gravity weight)
      const gsAdaptDt2 = body.gravityScale * gravWeight * dt2;
      body.position.x = body.prevPosition.x + body.velocity.x * dt + gravity.x * gsAdaptDt2;
      body.position.y = body.prevPosition.y + body.velocity.y * dt + gravity.y * gsAdaptDt2;
      body.position.z = body.prevPosition.z + body.velocity.z * dt + gravity.z * gsAdaptDt2;
    }

    const tBI = performance.now();

    // 5. Solver iterations
    const totalIters = config.postStabilize ? config.iterations + 1 : config.iterations;
    for (let iter = 0; iter < totalIters; iter++) {
      const isStab = config.postStabilize && iter === totalIters - 1;

      // Primal update
      for (const body of bodies) {
        if (body.type === RigidBodyType.Fixed) continue;
        this.primalUpdate3D(body, dt, isStab);
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

    const tEnd = performance.now();
    this.lastTimings = {
      total: tEnd - t0,
      broadphase: tBP - t0,
      narrowphase: tNP - tBP,
      warmstart: tWS - tNP,
      bodyInit: tBI - tWS,
      solverIters: tEnd - tBI,
      velocityRecover: 0,
      numBodies: bodies.length,
      numConstraints: this.constraintRows.length,
    };
  }

  private primalUpdate3D(body: Body3D, dt: number, isStabilization = false): void {
    const dt2 = dt * dt;
    const n = 6;

    // LHS = M/dt^2 (6x6 diagonal mass matrix) — reuse pooled array
    const lhs = this._lhs;
    lhs.fill(0);
    lhs[0 * n + 0] = body.mass / dt2;
    lhs[1 * n + 1] = body.mass / dt2;
    lhs[2 * n + 2] = body.mass / dt2;
    lhs[3 * n + 3] = body.inertia.x / dt2;
    lhs[4 * n + 4] = body.inertia.y / dt2;
    lhs[5 * n + 5] = body.inertia.z / dt2;

    // Position error for RHS (inline to avoid vec3Sub allocation)
    const dpx = body.position.x - body.inertialPosition.x;
    const dpy = body.position.y - body.inertialPosition.y;
    const dpz = body.position.z - body.inertialPosition.z;
    // Rotation error: small angle approximation (inline quatMul)
    const irw = body.inertialRotation.w, irx = -body.inertialRotation.x;
    const iry = -body.inertialRotation.y, irz = -body.inertialRotation.z;
    const rw = body.rotation.w, rx = body.rotation.x, ry = body.rotation.y, rz = body.rotation.z;
    const dqx = rw * irx + rx * irw + ry * irz - rz * iry;
    const dqy = rw * iry - rx * irz + ry * irw + rz * irx;
    const dqz = rw * irz + rx * iry - ry * irx + rz * irw;
    const dthetaX = 2 * dqx, dthetaY = 2 * dqy, dthetaZ = 2 * dqz;

    const mdt2 = body.mass / dt2;
    const rhs = this._rhs;
    rhs[0] = mdt2 * dpx;
    rhs[1] = mdt2 * dpy;
    rhs[2] = mdt2 * dpz;
    rhs[3] = body.inertia.x / dt2 * dthetaX;
    rhs[4] = body.inertia.y / dt2 * dthetaY;
    rhs[5] = body.inertia.z / dt2 * dthetaZ;

    // Accumulate constraints
    for (const row of this.constraintRows) {
      if (!row.active || row.broken) continue;

      let J: number[] | null = null;
      if (row.bodyA === body.index) J = row.jacobianA;
      else if (row.bodyB === body.index) J = row.jacobianB;
      else continue;

      // Evaluate linearized constraint: C = C0*(1-alpha) + J*dp
      // Per-iteration alpha (reference: solver.cpp):
      //   Normal iterations: alpha=1.0 → C0 term vanishes, only J·dp
      //   Stabilization: alpha=0.0 → full C0 correction
      const iterAlpha = isStabilization ? 0.0 : 1.0;
      let cEval = row.c0 * (1 - iterAlpha);
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

      // For soft constraints (finite stiffness), zero lambda in primal update.
      // Only hard constraints (infinite stiffness) use warmstarted lambda.
      // Reference: solver.cpp "lambda = isinf(stiffness[i]) ? force->lambda[i] : 0.0f"
      const lambdaForPrimal = isFinite(row.stiffness) ? 0 : row.lambda;
      let f = row.penalty * cEval + lambdaForPrimal;
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
    // Dual only runs on non-stabilization iterations, so alpha=1.0 → C0 term vanishes
    let cEval = 0; // C0*(1-1.0) = 0
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
    // For normal contacts: ramp when active. For friction: ramp when not sliding.
    // Reference: manifold.cpp — penalty += beta * |C| when active/sticking
    if (row.lambda > row.fmin && row.lambda < row.fmax) {
      row.penalty += this.config.beta * Math.abs(cEval);
    }
    row.penalty = Math.max(this.config.penaltyMin, Math.min(this.config.penaltyMax, row.penalty));
    if (row.penalty > row.stiffness) row.penalty = row.stiffness;
  }
}
