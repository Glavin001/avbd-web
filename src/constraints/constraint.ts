/**
 * Constraint system for the AVBD solver.
 * Each constraint has one or more "rows" — each row is a scalar constraint
 * with its own Jacobian, lambda, penalty, and force bounds.
 *
 * Ported from avbd-demo2d's Force class hierarchy.
 */

import type { Vec2 } from '../core/types.js';
import type { Body2D } from '../core/rigid-body.js';
import { ForceType } from '../core/types.js';

/**
 * A single constraint row for the AVBD solver.
 * Represents one scalar constraint equation C(x) = 0.
 */
export interface ConstraintRow {
  /** Index of body A (-1 for world anchor) */
  bodyA: number;
  /** Index of body B (-1 for world anchor) */
  bodyB: number;
  /** Force type identifier */
  type: ForceType;

  /** Jacobian for body A: dC/dx_A (3 components: dx, dy, dtheta) */
  jacobianA: [number, number, number];
  /** Jacobian for body B: dC/dx_B */
  jacobianB: [number, number, number];
  /** Diagonal of the Hessian for geometric stiffness (body A) */
  hessianDiagA: [number, number, number];
  /** Diagonal of the Hessian for geometric stiffness (body B) */
  hessianDiagB: [number, number, number];

  /** Current constraint value C(x) */
  c: number;
  /** Initial constraint value at start of iteration (for Taylor-series linearization) */
  c0: number;
  /** Stabilization alpha for this constraint */
  alpha: number;

  /** Lagrange multiplier (dual variable) */
  lambda: number;
  /** Penalty parameter (augmented Lagrangian) */
  penalty: number;
  /** Material stiffness (caps penalty) */
  stiffness: number;

  /** Minimum force bound */
  fmin: number;
  /** Maximum force bound */
  fmax: number;
  /** Fracture threshold (0 = never breaks) */
  fractureThreshold: number;

  /** Whether this constraint is active */
  active: boolean;
  /** Whether this constraint has been broken by fracture */
  broken: boolean;
}

export function createDefaultRow(): ConstraintRow {
  return {
    bodyA: -1,
    bodyB: -1,
    type: ForceType.Contact,
    jacobianA: [0, 0, 0],
    jacobianB: [0, 0, 0],
    hessianDiagA: [0, 0, 0],
    hessianDiagB: [0, 0, 0],
    c: 0,
    c0: 0,
    alpha: 0,
    lambda: 0,
    penalty: 100,
    stiffness: Infinity,
    fmin: -Infinity,
    fmax: Infinity,
    fractureThreshold: 0,
    active: true,
    broken: false,
  };
}

// ─── Constraint Store ───────────────────────────────────────────────────────

/**
 * Cached contact data for warmstarting between frames.
 * Stores lambda/penalty from previous contacts so they can be
 * transferred to new contacts at similar positions.
 */
export interface CachedContact {
  bodyA: number;
  bodyB: number;
  /** Normal lambda from previous frame */
  normalLambda: number;
  /** Normal penalty from previous frame */
  normalPenalty: number;
  /** Friction lambda from previous frame */
  frictionLambda: number;
  /** Friction penalty from previous frame */
  frictionPenalty: number;
  /** Contact position hash for matching */
  positionHash: number;
  /** Age in frames (for expiry) */
  age: number;
}

/** Simple spatial hash for contact position matching */
function contactHash(bodyA: number, bodyB: number, posX: number, posY: number): number {
  // Quantize position to grid cells for fuzzy matching
  const gx = Math.round(posX * 10);
  const gy = Math.round(posY * 10);
  return bodyA * 1000000 + bodyB * 10000 + gx * 100 + gy;
}

export class ConstraintStore {
  rows: ConstraintRow[] = [];

  /** Cache of contact data from previous frame for warmstarting */
  contactCache: Map<string, CachedContact> = new Map();

  /** Maximum age (in frames) before cached contacts expire */
  maxContactAge: number = 5;

  addRow(row: ConstraintRow): number {
    const index = this.rows.length;
    this.rows.push(row);
    return index;
  }

  addRows(rows: ConstraintRow[]): number[] {
    const indices: number[] = [];
    for (const row of rows) {
      indices.push(this.addRow(row));
    }
    return indices;
  }

  clear(): void {
    this.rows = [];
    this.contactCache.clear();
  }

  /**
   * Save current contact lambdas/penalties to the cache before clearing.
   * This enables warmstarting: new contacts at similar positions
   * inherit the multipliers from previous contacts.
   */
  cacheContacts(): void {
    // Age existing cache entries
    for (const [key, cached] of this.contactCache) {
      cached.age++;
      if (cached.age > this.maxContactAge) {
        this.contactCache.delete(key);
      }
    }

    // Save current contacts. Find normal rows (Contact type with infinite fmin)
    // and pair with the adjacent friction row. This avoids stride misalignment
    // when joint rows precede contact rows.
    for (let i = 0; i < this.rows.length; i++) {
      const row = this.rows[i];
      // Normal contact rows have type=Contact and fmin=-Infinity
      if (row.type !== ForceType.Contact || isFinite(row.fmin)) continue;
      if (i + 1 >= this.rows.length) break;

      const frictionRow = this.rows[i + 1];
      if (frictionRow.type !== ForceType.Contact) continue;

      const key = row.bodyA < row.bodyB
        ? `${row.bodyA}-${row.bodyB}`
        : `${row.bodyB}-${row.bodyA}`;

      this.contactCache.set(key, {
        bodyA: row.bodyA,
        bodyB: row.bodyB,
        normalLambda: row.lambda,
        normalPenalty: row.penalty,
        frictionLambda: frictionRow.lambda,
        frictionPenalty: frictionRow.penalty,
        positionHash: 0,
        age: 0,
      });
      i++; // Skip the friction row we just processed
    }
  }

  /**
   * Apply cached warmstart values to newly created contact rows.
   */
  warmstartContacts(): void {
    // Find normal rows (Contact type with infinite fmin) and pair with adjacent friction row.
    for (let i = 0; i < this.rows.length; i++) {
      const row = this.rows[i];
      // Normal contact rows have type=Contact and fmin=-Infinity
      if (row.type !== ForceType.Contact || isFinite(row.fmin)) continue;
      if (i + 1 >= this.rows.length) break;

      const frictionRow = this.rows[i + 1];
      if (frictionRow.type !== ForceType.Contact) continue;

      const key = row.bodyA < row.bodyB
        ? `${row.bodyA}-${row.bodyB}`
        : `${row.bodyB}-${row.bodyA}`;

      const cached = this.contactCache.get(key);
      if (cached) {
        row.lambda = cached.normalLambda;
        row.penalty = cached.normalPenalty;
        frictionRow.lambda = cached.frictionLambda;
        frictionRow.penalty = cached.frictionPenalty;
      }
      i++; // Skip the friction row we just processed
    }
  }

  clearContacts(): void {
    // Cache before clearing
    this.cacheContacts();
    // Remove all contact constraints, keep joints/springs/motors
    this.rows = this.rows.filter(r => r.type !== ForceType.Contact);
  }

  /** Get constraint row indices involving a specific body */
  getRowsForBody(bodyIndex: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < this.rows.length; i++) {
      const row = this.rows[i];
      if (row.active && (row.bodyA === bodyIndex || row.bodyB === bodyIndex)) {
        result.push(i);
      }
    }
    return result;
  }

  /** Get all active constraint pairs (for graph coloring) */
  getConstraintPairs(): [number, number][] {
    const pairs: [number, number][] = [];
    const seen = new Set<string>();
    for (const row of this.rows) {
      if (!row.active) continue;
      if (row.bodyA < 0 || row.bodyB < 0) continue;
      const key = row.bodyA < row.bodyB
        ? `${row.bodyA}-${row.bodyB}`
        : `${row.bodyB}-${row.bodyA}`;
      if (!seen.has(key)) {
        seen.add(key);
        pairs.push([row.bodyA, row.bodyB]);
      }
    }
    return pairs;
  }

  get count(): number {
    return this.rows.length;
  }

  get activeCount(): number {
    return this.rows.filter(r => r.active).length;
  }
}
