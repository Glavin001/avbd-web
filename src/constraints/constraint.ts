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

export class ConstraintStore {
  rows: ConstraintRow[] = [];

  /** Map from body pair key to constraint row indices */
  private contactMap: Map<string, number[]> = new Map();

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
    this.contactMap.clear();
  }

  clearContacts(): void {
    // Remove all contact constraints, keep joints/springs
    this.rows = this.rows.filter(r => r.type !== ForceType.Contact);
    this.contactMap.clear();
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
