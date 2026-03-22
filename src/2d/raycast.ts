/**
 * 2D Ray casting for the AVBD physics engine.
 * Supports ray-box and ray-circle intersection tests.
 */

import type { Vec2 } from '../core/types.js';
import type { Body2D } from '../core/rigid-body.js';
import { ColliderShapeType, type BodyStore2D } from '../core/rigid-body.js';
import { vec2, vec2Sub, vec2Dot, vec2Scale, vec2Add, vec2Length } from '../core/math.js';

export interface RaycastHit {
  /** Body that was hit */
  bodyIndex: number;
  /** Hit point in world space */
  point: Vec2;
  /** Surface normal at hit point */
  normal: Vec2;
  /** Parameter along the ray (0 = origin, 1 = origin + direction) */
  fraction: number;
}

/**
 * Cast a ray against all bodies in the world.
 * Returns the closest hit, or null if no intersection.
 */
export function raycastClosest(
  bodyStore: BodyStore2D,
  origin: Vec2,
  direction: Vec2,
  maxDistance: number = Infinity,
): RaycastHit | null {
  let closest: RaycastHit | null = null;
  let closestFraction = maxDistance / vec2Length(direction);

  for (const body of bodyStore.bodies) {
    if (body.boundingRadius <= 0) continue;

    const hit = raycastBody(body, origin, direction);
    if (hit && hit.fraction < closestFraction && hit.fraction >= 0) {
      closest = hit;
      closestFraction = hit.fraction;
    }
  }

  return closest;
}

/**
 * Cast a ray against all bodies, returning all hits sorted by distance.
 */
export function raycastAll(
  bodyStore: BodyStore2D,
  origin: Vec2,
  direction: Vec2,
  maxDistance: number = Infinity,
): RaycastHit[] {
  const hits: RaycastHit[] = [];
  const maxFraction = maxDistance / vec2Length(direction);

  for (const body of bodyStore.bodies) {
    if (body.boundingRadius <= 0) continue;

    const hit = raycastBody(body, origin, direction);
    if (hit && hit.fraction >= 0 && hit.fraction <= maxFraction) {
      hits.push(hit);
    }
  }

  hits.sort((a, b) => a.fraction - b.fraction);
  return hits;
}

/**
 * Cast a ray against a single body.
 */
function raycastBody(body: Body2D, origin: Vec2, direction: Vec2): RaycastHit | null {
  if (body.colliderShape === ColliderShapeType.Ball) {
    return raycastCircle(body, origin, direction);
  }
  return raycastBox(body, origin, direction);
}

/**
 * Ray-circle intersection test.
 */
function raycastCircle(body: Body2D, origin: Vec2, dir: Vec2): RaycastHit | null {
  const oc = vec2Sub(origin, body.position);
  const a = vec2Dot(dir, dir);
  const b = 2 * vec2Dot(oc, dir);
  const c = vec2Dot(oc, oc) - body.radius * body.radius;
  const discriminant = b * b - 4 * a * c;

  if (discriminant < 0) return null;

  const sqrtD = Math.sqrt(discriminant);
  let t = (-b - sqrtD) / (2 * a);

  // If t < 0, try the other intersection (ray starts inside circle)
  if (t < 0) {
    t = (-b + sqrtD) / (2 * a);
    if (t < 0) return null;
  }

  const point = vec2Add(origin, vec2Scale(dir, t));
  const normal = vec2Sub(point, body.position);
  const normalLen = vec2Length(normal);

  return {
    bodyIndex: body.index,
    point,
    normal: normalLen > 1e-10
      ? vec2Scale(normal, 1 / normalLen)
      : vec2(0, 1),
    fraction: t,
  };
}

/**
 * Ray-OBB (oriented bounding box) intersection using slab method.
 */
function raycastBox(body: Body2D, origin: Vec2, dir: Vec2): RaycastHit | null {
  // Transform ray to box local space
  const cosA = Math.cos(-body.angle);
  const sinA = Math.sin(-body.angle);
  const d = vec2Sub(origin, body.position);

  const localOrigin: Vec2 = {
    x: cosA * d.x - sinA * d.y,
    y: sinA * d.x + cosA * d.y,
  };
  const localDir: Vec2 = {
    x: cosA * dir.x - sinA * dir.y,
    y: sinA * dir.x + cosA * dir.y,
  };

  // Slab test for AABB in local space
  const hx = body.halfExtents.x;
  const hy = body.halfExtents.y;

  let tmin = -Infinity;
  let tmax = Infinity;
  let normalLocal: Vec2 = vec2(0, 0);

  // X slab
  if (Math.abs(localDir.x) > 1e-10) {
    let t1 = (-hx - localOrigin.x) / localDir.x;
    let t2 = (hx - localOrigin.x) / localDir.x;
    let n1 = vec2(-1, 0);
    let n2 = vec2(1, 0);
    if (t1 > t2) { [t1, t2] = [t2, t1]; [n1, n2] = [n2, n1]; }
    if (t1 > tmin) { tmin = t1; normalLocal = n1; }
    if (t2 < tmax) { tmax = t2; }
  } else if (localOrigin.x < -hx || localOrigin.x > hx) {
    return null;
  }

  // Y slab
  if (Math.abs(localDir.y) > 1e-10) {
    let t1 = (-hy - localOrigin.y) / localDir.y;
    let t2 = (hy - localOrigin.y) / localDir.y;
    let n1 = vec2(0, -1);
    let n2 = vec2(0, 1);
    if (t1 > t2) { [t1, t2] = [t2, t1]; [n1, n2] = [n2, n1]; }
    if (t1 > tmin) { tmin = t1; normalLocal = n1; }
    if (t2 < tmax) { tmax = t2; }
  } else if (localOrigin.y < -hy || localOrigin.y > hy) {
    return null;
  }

  if (tmin > tmax || tmax < 0) return null;

  const t = tmin >= 0 ? tmin : tmax;
  if (t < 0) return null;

  const point = vec2Add(origin, vec2Scale(dir, t));

  // Transform normal back to world space
  const cosB = Math.cos(body.angle);
  const sinB = Math.sin(body.angle);
  const worldNormal: Vec2 = {
    x: cosB * normalLocal.x - sinB * normalLocal.y,
    y: sinB * normalLocal.x + cosB * normalLocal.y,
  };

  return {
    bodyIndex: body.index,
    point,
    normal: worldNormal,
    fraction: t,
  };
}
