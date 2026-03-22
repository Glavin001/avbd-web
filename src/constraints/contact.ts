/**
 * Contact constraint generation from collision manifolds.
 * Implements the Taylor-series linearization from AVBD (Sec 4 of VBD paper).
 *
 * Each contact point generates 2 constraint rows:
 * - Normal constraint (non-penetration, unilateral)
 * - Friction constraint (bilateral with Coulomb limit: |f| <= mu * |f_normal|)
 */

import type { Vec2, ContactManifold2D, ContactPoint2D } from '../core/types.js';
import type { Body2D } from '../core/rigid-body.js';
import { ForceType } from '../core/types.js';
import { createDefaultRow, type ConstraintRow } from './constraint.js';
import { vec2Sub, vec2Dot, vec2Cross, vec2Perp } from '../core/math.js';

/**
 * Compute the Jacobian for a 2D rigid body contact.
 * J_A = [n.x, n.y, (r_A × n)] where r_A = p - x_A
 */
function computeContactJacobian(
  bodyPosition: Vec2,
  contactPoint: Vec2,
  normal: Vec2,
): [number, number, number] {
  const r = vec2Sub(contactPoint, bodyPosition);
  const torque = vec2Cross(r, normal);
  return [normal.x, normal.y, torque];
}

/**
 * Compute relative normal velocity at contact for restitution.
 */
function computeRelativeNormalVelocity(
  bodyA: Body2D,
  bodyB: Body2D,
  contactPoint: Vec2,
  normal: Vec2,
): number {
  const rA = vec2Sub(contactPoint, bodyA.position);
  const rB = vec2Sub(contactPoint, bodyB.position);
  // Point velocity = v + ω × r
  const vA = {
    x: bodyA.velocity.x - bodyA.angularVelocity * rA.y,
    y: bodyA.velocity.y + bodyA.angularVelocity * rA.x,
  };
  const vB = {
    x: bodyB.velocity.x - bodyB.angularVelocity * rB.y,
    y: bodyB.velocity.y + bodyB.angularVelocity * rB.x,
  };
  const relVel = { x: vA.x - vB.x, y: vA.y - vB.y };
  return vec2Dot(relVel, normal);
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
  const mu = Math.sqrt(bodyA.friction * bodyB.friction);
  const restitution = Math.max(bodyA.restitution, bodyB.restitution);

  for (const contact of manifold.contacts) {
    // ─── Normal constraint row ──────────────────────────────────
    const normalRow = createDefaultRow();
    normalRow.bodyA = manifold.bodyA;
    normalRow.bodyB = manifold.bodyB;
    normalRow.type = ForceType.Contact;

    const n = manifold.normal;

    normalRow.jacobianA = computeContactJacobian(bodyA.position, contact.position, n);
    normalRow.jacobianB = computeContactJacobian(bodyB.position, contact.position, n);
    normalRow.jacobianB[0] = -normalRow.jacobianB[0];
    normalRow.jacobianB[1] = -normalRow.jacobianB[1];
    normalRow.jacobianB[2] = -normalRow.jacobianB[2];

    // C < 0 when penetrating, C = 0 at contact surface
    normalRow.c = -contact.depth;
    normalRow.c0 = normalRow.c;

    // Apply restitution: modify the constraint target to include bounce velocity
    // v_bounce = -e * v_n (if approaching)
    if (restitution > 0) {
      const vn = computeRelativeNormalVelocity(bodyA, bodyB, contact.position, n);
      if (vn < -0.5) { // Only apply restitution above a velocity threshold
        // Bias the constraint to target a separating velocity
        // c0 is modified to account for the desired post-collision velocity
        const dt = 1 / 60; // Will be overridden by solver
        normalRow.c0 = normalRow.c + restitution * vn * dt;
      }
    }

    // Compressive force only
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

    fricRow.c = 0;
    fricRow.c0 = 0;

    // Initialize friction bounds from estimated normal force (penalty * depth)
    // This gets corrected during dual update with actual normal lambda
    const estimatedNormalForce = penaltyMin * contact.depth;
    fricRow.fmin = -mu * estimatedNormalForce;
    fricRow.fmax = mu * estimatedNormalForce;

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
