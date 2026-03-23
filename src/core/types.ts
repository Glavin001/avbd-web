/**
 * Core types for the AVBD WebGPU physics engine.
 * Supports both 2D (3-DOF) and 3D (6-DOF) modes.
 */

// ─── Vector Types ───────────────────────────────────────────────────────────

export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quat {
  w: number;
  x: number;
  y: number;
  z: number;
}

// ─── Dimension Mode ─────────────────────────────────────────────────────────

export type DimensionMode = '2d' | '3d';

// ─── Rigid Body Types ───────────────────────────────────────────────────────

export enum RigidBodyType {
  Dynamic = 0,
  Fixed = 1,
  KinematicPositionBased = 2,
  KinematicVelocityBased = 3,
}

export interface RigidBodyHandle {
  readonly index: number;
}

/**
 * 2D rigid body state: [x, y, angle, vx, vy, omega, mass, inertia]
 * Stored as flat Float32Array for GPU upload.
 */
export const BODY_STATE_2D_STRIDE = 8;

/**
 * 3D rigid body state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, mass, Ixx, Iyy, Izz]
 * Stored as flat Float32Array for GPU upload.
 */
export const BODY_STATE_3D_STRIDE = 20;

// ─── Collider Types ─────────────────────────────────────────────────────────

export enum ColliderShape {
  Cuboid = 0,
  Ball = 1,
}

export interface ColliderHandle {
  readonly index: number;
}

// ─── Force / Constraint Types ───────────────────────────────────────────────

export enum ForceType {
  Contact = 0,
  Joint = 1,
  Spring = 2,
  Motor = 3,
}

/**
 * Per-row constraint data for GPU buffer.
 * Each constraint "row" has:
 * - bodyA index, bodyB index
 * - Jacobian A (3 or 6 floats), Jacobian B (3 or 6 floats)
 * - constraint value C, initial constraint value C0
 * - lambda (Lagrange multiplier)
 * - penalty (augmented Lagrangian penalty parameter)
 * - stiffness (material stiffness, caps penalty)
 * - fmin, fmax (force bounds)
 * - fracture threshold
 */
export const CONSTRAINT_ROW_2D_STRIDE = 24;
// bodyA(1) + bodyB(1) + JA(3) + JB(3) + C(1) + C0(1) + lambda(1) + penalty(1)
// + stiffness(1) + fmin(1) + fmax(1) + fracture(1) + alpha(1) + type(1)
// + hessA_diag(3) + hessB_diag(3) = 24

// ─── Solver Configuration ───────────────────────────────────────────────────

export interface SolverConfig {
  /** Solver iterations per timestep (default: 10) */
  iterations: number;
  /** Timestep in seconds (default: 1/60) */
  dt: number;
  /** Penalty growth rate (default: 100000) */
  beta: number;
  /** Stabilization parameter (default: 0.99) */
  alpha: number;
  /** Warmstart decay (default: 0.99) */
  gamma: number;
  /** Enable post-stabilization iteration (default: true) */
  postStabilize: boolean;
  /** Minimum penalty parameter (default: 100) */
  penaltyMin: number;
  /** Maximum penalty parameter (default: 1e10) */
  penaltyMax: number;
  /** Gravity vector */
  gravity: Vec2 | Vec3;
}

export const DEFAULT_SOLVER_CONFIG_2D: SolverConfig = {
  iterations: 10,
  dt: 1 / 60,
  beta: 100000,
  alpha: 0.99,
  gamma: 0.99,
  postStabilize: true,
  penaltyMin: 100,
  penaltyMax: 1e9,
  gravity: { x: 0, y: -9.81 },
};

export const DEFAULT_SOLVER_CONFIG_3D: SolverConfig = {
  iterations: 10,
  dt: 1 / 60,
  beta: 100000,
  alpha: 0.99,
  gamma: 0.99,
  postStabilize: true,
  penaltyMin: 100,
  penaltyMax: 1e9,
  gravity: { x: 0, y: -9.81, z: 0 },
};

// ─── Graph Coloring ─────────────────────────────────────────────────────────

export interface ColorGroup {
  /** Color index */
  color: number;
  /** Body indices in this color group */
  bodyIndices: number[];
}

// ─── Contact / Collision Types ──────────────────────────────────────────────

export interface ContactPoint2D {
  /** Contact position in world space */
  position: Vec2;
  /** Contact normal (from B to A) */
  normal: Vec2;
  /** Penetration depth (positive = overlapping) */
  depth: number;
}

export interface ContactManifold2D {
  bodyA: number;
  bodyB: number;
  normal: Vec2;
  contacts: ContactPoint2D[];
}

// ─── Material Properties ────────────────────────────────────────────────────

export interface MaterialProperties {
  friction: number;
  restitution: number;
  density: number;
}

/** Collision margin added to normal contacts (reference: 0.0005) */
export const COLLISION_MARGIN = 0.0005;

export const DEFAULT_MATERIAL: MaterialProperties = {
  friction: 0.5,
  restitution: 0.3,
  density: 1.0,
};
