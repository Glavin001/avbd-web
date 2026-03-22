/**
 * Distance spring constraint for the AVBD solver.
 * Constrains two bodies to maintain a target distance.
 * Ported from avbd-demo2d/source/spring.cpp.
 */

import type { Vec2 } from '../core/types.js';
import type { Body2D } from '../core/rigid-body.js';
import { ForceType } from '../core/types.js';
import { createDefaultRow, type ConstraintRow } from './constraint.js';
import { vec2Sub, vec2Length, vec2Rotate, vec2Add } from '../core/math.js';

export interface SpringDef2D {
  bodyA: number;
  bodyB: number;
  localAnchorA: Vec2;
  localAnchorB: Vec2;
  restLength: number;
  stiffness: number;
  damping: number;
}

/**
 * Create constraint rows for a distance spring.
 * Generates 1 row: the signed distance error.
 */
export function createSpringConstraintRows(
  spring: SpringDef2D,
  bodyA: Body2D,
  bodyB: Body2D,
  penaltyMin: number,
): ConstraintRow[] {
  // World-space anchor positions
  const anchorA = vec2Add(bodyA.position, vec2Rotate(spring.localAnchorA, bodyA.angle));
  const anchorB = vec2Add(bodyB.position, vec2Rotate(spring.localAnchorB, bodyB.angle));

  const rA = vec2Rotate(spring.localAnchorA, bodyA.angle);
  const rB = vec2Rotate(spring.localAnchorB, bodyB.angle);

  const diff = vec2Sub(anchorA, anchorB);
  const dist = vec2Length(diff);

  if (dist < 1e-10) return [];

  // Unit direction from B to A
  const nx = diff.x / dist;
  const ny = diff.y / dist;

  // Constraint: C = dist - restLength
  const c = dist - spring.restLength;

  const row = createDefaultRow();
  row.bodyA = spring.bodyA;
  row.bodyB = spring.bodyB;
  row.type = ForceType.Spring;

  // Jacobian: dC/dx = direction of stretching
  // d(dist)/dx_A = n (unit direction A→B)
  // d(dist)/dy_A = same for y
  // d(dist)/dtheta_A = rA × n
  row.jacobianA = [nx, ny, rA.x * ny - rA.y * nx];
  row.jacobianB = [-nx, -ny, -(rB.x * ny - rB.y * nx)];

  // Hessian diagonal (second derivative of distance constraint)
  row.hessianDiagA = [0, 0, -(rA.x * nx + rA.y * ny)];
  row.hessianDiagB = [0, 0, -(rB.x * nx + rB.y * ny)];

  row.c = c;
  row.c0 = c;
  row.stiffness = spring.stiffness;
  row.penalty = Math.min(penaltyMin, spring.stiffness);

  return [row];
}
