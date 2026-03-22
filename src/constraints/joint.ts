/**
 * Joint constraints for the AVBD solver.
 * Ported from avbd-demo2d/source/joint.cpp.
 *
 * Revolute joint: constrains two bodies to share a point.
 * Generates 2 rows (x and y components) + optional angle row.
 */

import type { Vec2 } from '../core/types.js';
import type { Body2D } from '../core/rigid-body.js';
import { ForceType } from '../core/types.js';
import { createDefaultRow, type ConstraintRow } from './constraint.js';
import { vec2Rotate, vec2Add, vec2Sub } from '../core/math.js';

export interface JointDef2D {
  bodyA: number;
  bodyB: number;
  /** Anchor point on body A in local coordinates */
  localAnchorA: Vec2;
  /** Anchor point on body B in local coordinates */
  localAnchorB: Vec2;
  /** Optional angle constraint (for welded joints) */
  angleConstraint: boolean;
  /** Target relative angle (for welded/motor joints) */
  targetAngle: number;
  /** Joint stiffness (Infinity = rigid joint) */
  stiffness: number;
  /** Fracture threshold (0 = unbreakable) */
  fractureThreshold: number;
}

/**
 * Create constraint rows for a revolute joint.
 * A revolute joint constrains the world-space anchor points on both bodies to coincide.
 *
 * Row 0: x-component of position constraint
 * Row 1: y-component of position constraint
 * Row 2 (optional): angle constraint
 */
export function createJointConstraintRows(
  joint: JointDef2D,
  bodyA: Body2D,
  bodyB: Body2D,
  penaltyMin: number,
): ConstraintRow[] {
  const rows: ConstraintRow[] = [];

  // World-space anchor positions
  const anchorA = vec2Add(bodyA.position, vec2Rotate(joint.localAnchorA, bodyA.angle));
  const anchorB = vec2Add(bodyB.position, vec2Rotate(joint.localAnchorB, bodyB.angle));

  // Rotated local anchors (needed for Jacobian)
  const rA = vec2Rotate(joint.localAnchorA, bodyA.angle);
  const rB = vec2Rotate(joint.localAnchorB, bodyB.angle);

  // Position error
  const error = vec2Sub(anchorA, anchorB);

  // ─── X constraint ──────────────────────────────────────────────
  const rowX = createDefaultRow();
  rowX.bodyA = joint.bodyA;
  rowX.bodyB = joint.bodyB;
  rowX.type = ForceType.Joint;

  // d(anchor_A.x)/d(x_A) = 1
  // d(anchor_A.x)/d(y_A) = 0
  // d(anchor_A.x)/d(theta_A) = -rA.y (derivative of rotation)
  rowX.jacobianA = [1, 0, -rA.y];
  rowX.jacobianB = [-1, 0, rB.y];

  rowX.c = error.x;
  rowX.c0 = error.x;
  rowX.stiffness = joint.stiffness;
  rowX.penalty = penaltyMin;
  rowX.fractureThreshold = joint.fractureThreshold;

  // Hessian diagonal for geometric stiffness
  // d²(anchor_A.x)/d(theta_A)² = -rA.x
  rowX.hessianDiagA = [0, 0, -rA.x];
  rowX.hessianDiagB = [0, 0, -rB.x];

  rows.push(rowX);

  // ─── Y constraint ──────────────────────────────────────────────
  const rowY = createDefaultRow();
  rowY.bodyA = joint.bodyA;
  rowY.bodyB = joint.bodyB;
  rowY.type = ForceType.Joint;

  // d(anchor_A.y)/d(x_A) = 0
  // d(anchor_A.y)/d(y_A) = 1
  // d(anchor_A.y)/d(theta_A) = rA.x
  rowY.jacobianA = [0, 1, rA.x];
  rowY.jacobianB = [0, -1, -rB.x];

  rowY.c = error.y;
  rowY.c0 = error.y;
  rowY.stiffness = joint.stiffness;
  rowY.penalty = penaltyMin;
  rowY.fractureThreshold = joint.fractureThreshold;

  rowY.hessianDiagA = [0, 0, -rA.y];
  rowY.hessianDiagB = [0, 0, -rB.y];

  rows.push(rowY);

  // ─── Angle constraint (optional) ───────────────────────────────
  if (joint.angleConstraint) {
    const rowAngle = createDefaultRow();
    rowAngle.bodyA = joint.bodyA;
    rowAngle.bodyB = joint.bodyB;
    rowAngle.type = ForceType.Joint;

    // d(theta_A - theta_B - target)/d(theta_A) = 1
    // d(theta_A - theta_B - target)/d(theta_B) = -1
    rowAngle.jacobianA = [0, 0, 1];
    rowAngle.jacobianB = [0, 0, -1];

    let angleDiff = bodyA.angle - bodyB.angle - joint.targetAngle;
    // Normalize to [-pi, pi]
    while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

    rowAngle.c = angleDiff;
    rowAngle.c0 = angleDiff;
    rowAngle.stiffness = joint.stiffness;
    rowAngle.penalty = penaltyMin;
    rowAngle.fractureThreshold = joint.fractureThreshold;

    rows.push(rowAngle);
  }

  return rows;
}

// ─── JointData (Rapier-style API) ───────────────────────────────────────────

export class JointData2D {
  localAnchorA: Vec2;
  localAnchorB: Vec2;
  angleConstraint: boolean;
  targetAngle: number;
  stiffness: number;
  fractureThreshold: number;

  private constructor(localAnchorA: Vec2, localAnchorB: Vec2) {
    this.localAnchorA = localAnchorA;
    this.localAnchorB = localAnchorB;
    this.angleConstraint = false;
    this.targetAngle = 0;
    this.stiffness = Infinity;
    this.fractureThreshold = 0;
  }

  /** Create a revolute (pivot/hinge) joint */
  static revolute(anchorA: Vec2, anchorB: Vec2): JointData2D {
    return new JointData2D(anchorA, anchorB);
  }

  /** Create a fixed (welded) joint that also constrains angle */
  static fixed(anchorA: Vec2, anchorB: Vec2): JointData2D {
    const j = new JointData2D(anchorA, anchorB);
    j.angleConstraint = true;
    return j;
  }

  setStiffness(stiffness: number): JointData2D {
    this.stiffness = stiffness;
    return this;
  }

  setFractureThreshold(threshold: number): JointData2D {
    this.fractureThreshold = threshold;
    return this;
  }
}
