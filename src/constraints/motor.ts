/**
 * Motor constraint for the AVBD solver.
 * Drives a body or joint to a target angular velocity.
 * Ported from avbd-demo2d/source/motor.cpp.
 */

import type { Body2D } from '../core/rigid-body.js';
import { ForceType } from '../core/types.js';
import { createDefaultRow, type ConstraintRow } from './constraint.js';

export interface MotorDef2D {
  bodyA: number;
  bodyB: number;
  /** Target angular velocity (rad/s) */
  targetVelocity: number;
  /** Maximum torque the motor can apply */
  maxTorque: number;
  /** Motor stiffness (how strongly it drives to target) */
  stiffness: number;
}

/**
 * Create constraint rows for an angular velocity motor.
 * The motor drives the relative angular velocity between two bodies.
 *
 * Constraint: C = (theta_A - theta_B) - target_velocity * dt
 * This creates a velocity-level constraint that drives relative rotation.
 */
export function createMotorConstraintRows(
  motor: MotorDef2D,
  bodyA: Body2D,
  bodyB: Body2D,
  penaltyMin: number,
  dt: number,
): ConstraintRow[] {
  const row = createDefaultRow();
  row.bodyA = motor.bodyA;
  row.bodyB = motor.bodyB;
  row.type = ForceType.Motor;

  // Angular-only Jacobian: d(theta_A - theta_B)/d(state)
  row.jacobianA = [0, 0, 1];
  row.jacobianB = [0, 0, -1];

  // The constraint value represents the angular velocity error
  // We want omega_A - omega_B = targetVelocity
  // Expressed as position constraint: theta_A - theta_B - targetVelocity * dt = 0
  const relAngle = bodyA.angle - bodyB.angle;
  const targetAngle = relAngle + motor.targetVelocity * dt;
  row.c = bodyA.angle - bodyB.angle - targetAngle;
  row.c0 = row.c;

  // Motor force bounds (capped by max torque)
  row.fmin = -motor.maxTorque;
  row.fmax = motor.maxTorque;

  row.penalty = penaltyMin;
  row.stiffness = motor.stiffness;

  return [row];
}

// ─── MotorData (Rapier-style API) ───────────────────────────────────────────

export class MotorData2D {
  targetVelocity: number;
  maxTorque: number;
  stiffness: number;

  private constructor(targetVelocity: number) {
    this.targetVelocity = targetVelocity;
    this.maxTorque = Infinity;
    this.stiffness = Infinity;
  }

  /** Create an angular velocity motor */
  static velocity(targetVelocity: number): MotorData2D {
    return new MotorData2D(targetVelocity);
  }

  setMaxTorque(torque: number): MotorData2D {
    this.maxTorque = torque;
    return this;
  }

  setStiffness(stiffness: number): MotorData2D {
    this.stiffness = stiffness;
    return this;
  }
}
