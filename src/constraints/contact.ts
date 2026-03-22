/**
 * Contact constraint generation from collision manifolds.
 * Implements the Taylor-series linearization from AVBD (Sec 4 of VBD paper).
 *
 * Each contact point generates 2 constraint rows:
 * - Normal constraint (non-penetration, unilateral: fmin=0)
 * - Friction constraint (bilateral with Coulomb limit: |f| <= mu * |f_normal|)
 */

import type { Vec2, ContactManifold2D, ContactPoint2D } from '../core/types.js';
import type { Body2D } from '../core/rigid-body.js';
import { ForceType } from '../core/types.js';
import { createDefaultRow, type ConstraintRow } from './constraint.js';
import { vec2Sub, vec2Dot, vec2Cross, vec2Perp } from '../core/math.js';

/**
 * Compute the Jacobian for a 2D rigid body contact.
 * For body A at contact point p with normal n:
 * J_A = [n.x, n.y, (r_A × n)]
 * where r_A = p - x_A
 */
function computeContactJacobian(
  bodyPosition: Vec2,
  contactPoint: Vec2,
  normal: Vec2,
): [number, number, number] {
  const r = vec2Sub(contactPoint, bodyPosition);
  const torque = vec2Cross(r, normal); // r × n (scalar in 2D)
  return [normal.x, normal.y, torque];
}

/**
 * Create constraint rows from a contact manifold.
 * Each contact point generates a normal row and a friction row.
 */
export function createContactConstraintRows(
  manifold: ContactManifold2D,
  bodyA: Body2D,
  bodyB: Body2D,
  penaltyMin: number,
  stiffness: number = Infinity,
): ConstraintRow[] {
  const rows: ConstraintRow[] = [];
  const mu = Math.sqrt(bodyA.friction * bodyB.friction); // Geometric mean friction

  for (const contact of manifold.contacts) {
    // ─── Normal constraint row ──────────────────────────────────
    const normalRow = createDefaultRow();
    normalRow.bodyA = manifold.bodyA;
    normalRow.bodyB = manifold.bodyB;
    normalRow.type = ForceType.Contact;

    // Normal points from B to A
    const n = manifold.normal;

    // Jacobians: J_A = dC/dx_A, J_B = dC/dx_B = -dC/dx_A
    normalRow.jacobianA = computeContactJacobian(bodyA.position, contact.position, n);
    normalRow.jacobianB = computeContactJacobian(bodyB.position, contact.position, n);
    // Negate B's Jacobian (opposing contribution)
    normalRow.jacobianB[0] = -normalRow.jacobianB[0];
    normalRow.jacobianB[1] = -normalRow.jacobianB[1];
    normalRow.jacobianB[2] = -normalRow.jacobianB[2];

    // Constraint value: negative depth means penetration
    // C < 0 when penetrating, C = 0 at contact surface
    normalRow.c = -contact.depth;
    normalRow.c0 = normalRow.c;

    // Unilateral: compressive force only (force pushes bodies apart)
    // With C < 0 when penetrating and J pointing in separation direction,
    // the force f = clamp(penalty*C + lambda, fmin, fmax) must be <= 0
    normalRow.fmin = -Infinity;
    normalRow.fmax = 0;

    normalRow.penalty = penaltyMin;
    normalRow.stiffness = stiffness;
    normalRow.lambda = 0;

    rows.push(normalRow);

    // ─── Friction constraint row ────────────────────────────────
    const tangent: Vec2 = { x: -n.y, y: n.x };

    const fricRow = createDefaultRow();
    fricRow.bodyA = manifold.bodyA;
    fricRow.bodyB = manifold.bodyB;
    fricRow.type = ForceType.Contact;

    fricRow.jacobianA = computeContactJacobian(bodyA.position, contact.position, tangent);
    fricRow.jacobianB = computeContactJacobian(bodyB.position, contact.position, tangent);
    fricRow.jacobianB[0] = -fricRow.jacobianB[0];
    fricRow.jacobianB[1] = -fricRow.jacobianB[1];
    fricRow.jacobianB[2] = -fricRow.jacobianB[2];

    // Friction constraint value: relative tangential displacement
    // Initially zero (set up at contact creation)
    fricRow.c = 0;
    fricRow.c0 = 0;

    // Coulomb friction: |f_tangent| <= mu * |f_normal|
    // We'll dynamically update fmax during the dual update
    fricRow.fmin = -mu * penaltyMin * Math.abs(normalRow.c);
    fricRow.fmax = mu * penaltyMin * Math.abs(normalRow.c);

    fricRow.penalty = penaltyMin;
    fricRow.stiffness = stiffness;
    fricRow.lambda = 0;

    rows.push(fricRow);
  }

  return rows;
}

/**
 * Update friction bounds based on current normal force.
 * Called during dual update phase.
 */
export function updateFrictionBounds(
  normalRow: ConstraintRow,
  frictionRow: ConstraintRow,
  bodyA: Body2D,
  bodyB: Body2D,
): void {
  const mu = Math.sqrt(bodyA.friction * bodyB.friction);
  const normalForce = Math.abs(normalRow.lambda);
  frictionRow.fmin = -mu * normalForce;
  frictionRow.fmax = mu * normalForce;
}
