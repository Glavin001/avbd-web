/**
 * 2D collision detection for the AVBD physics engine.
 * Supports box-box, box-circle, and circle-circle collisions.
 *
 * Box-box uses SAT (Separating Axis Theorem) with Sutherland-Hodgman clipping.
 */

import type { Vec2, ContactManifold2D, ContactPoint2D } from '../core/types.js';
import type { Body2D } from '../core/rigid-body.js';
import { ColliderShapeType } from '../core/rigid-body.js';
import {
  vec2, vec2Add, vec2Sub, vec2Scale, vec2Dot, vec2Cross, vec2Length,
  vec2Normalize, vec2Neg,
} from '../core/math.js';

// ─── Helper: get box vertices and face normals ──────────────────────────────

function getBoxVertices(body: Body2D): Vec2[] {
  const c = Math.cos(body.angle);
  const s = Math.sin(body.angle);
  const hx = body.halfExtents.x;
  const hy = body.halfExtents.y;
  const cx = body.position.x;
  const cy = body.position.y;

  return [
    { x: cx + c * (-hx) - s * (-hy), y: cy + s * (-hx) + c * (-hy) },
    { x: cx + c * (hx) - s * (-hy), y: cy + s * (hx) + c * (-hy) },
    { x: cx + c * (hx) - s * (hy), y: cy + s * (hx) + c * (hy) },
    { x: cx + c * (-hx) - s * (hy), y: cy + s * (-hx) + c * (hy) },
  ];
}

/** Get outward-pointing face normals (perpendicular to each edge) */
function getBoxNormals(body: Body2D): Vec2[] {
  const c = Math.cos(body.angle);
  const s = Math.sin(body.angle);
  return [
    { x: c, y: s },    // right face normal
    { x: -s, y: c },   // top face normal
    { x: -c, y: -s },  // left face normal
    { x: s, y: -c },   // bottom face normal
  ];
}

/** Project polygon onto axis, return [min, max] */
function projectPolygon(vertices: Vec2[], axis: Vec2): [number, number] {
  let min = Infinity, max = -Infinity;
  for (const v of vertices) {
    const d = vec2Dot(axis, v);
    if (d < min) min = d;
    if (d > max) max = d;
  }
  return [min, max];
}

// ─── Box-Box SAT ────────────────────────────────────────────────────────────

export function collideBoxBox(bodyA: Body2D, bodyB: Body2D): ContactManifold2D | null {
  const vertsA = getBoxVertices(bodyA);
  const vertsB = getBoxVertices(bodyB);

  // Only need 2 unique normals per box (opposite faces share the same axis)
  const cA = Math.cos(bodyA.angle), sA = Math.sin(bodyA.angle);
  const cB = Math.cos(bodyB.angle), sB = Math.sin(bodyB.angle);
  const axes: Vec2[] = [
    { x: cA, y: sA },
    { x: -sA, y: cA },
    { x: cB, y: sB },
    { x: -sB, y: cB },
  ];

  let minOverlap = Infinity;
  let bestAxis: Vec2 = { x: 0, y: 0 };

  for (const axis of axes) {
    const [minA, maxA] = projectPolygon(vertsA, axis);
    const [minB, maxB] = projectPolygon(vertsB, axis);

    const overlap = Math.min(maxA - minB, maxB - minA);
    if (overlap < 0) return null; // Separating axis found

    if (overlap < minOverlap) {
      minOverlap = overlap;
      bestAxis = axis;
    }
  }

  // Ensure normal points from B to A
  const d = vec2Sub(bodyA.position, bodyB.position);
  if (vec2Dot(d, bestAxis) < 0) {
    bestAxis = vec2Neg(bestAxis);
  }

  // Find contact points using edge clipping
  const contacts = findContactPoints(vertsA, vertsB, bestAxis, minOverlap);

  if (contacts.length === 0) {
    // Fallback: use midpoint contact
    const mid = vec2Scale(vec2Add(bodyA.position, bodyB.position), 0.5);
    contacts.push({
      position: mid,
      normal: bestAxis,
      depth: minOverlap,
    });
  }

  return {
    bodyA: bodyA.index,
    bodyB: bodyB.index,
    normal: bestAxis,
    contacts,
  };
}

/**
 * Find contact points between two convex polygons given the collision normal.
 * Uses edge clipping (Sutherland-Hodgman style).
 */
function findContactPoints(
  vertsA: Vec2[],
  vertsB: Vec2[],
  normal: Vec2,
  depth: number,
): ContactPoint2D[] {
  // Find the support edges
  const edgeA = findSupportEdge(vertsA, vec2Neg(normal));
  const edgeB = findSupportEdge(vertsB, normal);

  // Determine reference and incident edges
  // Reference edge is more perpendicular to the collision normal
  const edgeADir = vec2Normalize(vec2Sub(edgeA[1], edgeA[0]));
  const edgeBDir = vec2Normalize(vec2Sub(edgeB[1], edgeB[0]));

  const dotA = Math.abs(vec2Dot(edgeADir, normal));
  const dotB = Math.abs(vec2Dot(edgeBDir, normal));

  let refEdge: [Vec2, Vec2];
  let incEdge: [Vec2, Vec2];
  let refNormal: Vec2;
  let flip: boolean;

  if (dotA <= dotB) {
    // Edge A is more perpendicular to normal — use as reference
    refEdge = edgeA;
    incEdge = edgeB;
    refNormal = normal;
    flip = false;
  } else {
    refEdge = edgeB;
    incEdge = edgeA;
    refNormal = vec2Neg(normal);
    flip = true;
  }

  // Clip incident edge against reference edge's side planes
  const refDir = vec2Normalize(vec2Sub(refEdge[1], refEdge[0]));
  const refLen = vec2Length(vec2Sub(refEdge[1], refEdge[0]));

  // Clip against left side
  let clipped = clipEdge(incEdge[0], incEdge[1], vec2Neg(refDir), -vec2Dot(vec2Neg(refDir), refEdge[0]));
  if (clipped.length < 2) return [];

  // Clip against right side
  clipped = clipEdge(clipped[0], clipped[1], refDir, -vec2Dot(refDir, refEdge[1]));
  if (clipped.length < 2) return [];

  // Keep only points behind the reference face
  const refFaceNormal = flip ? vec2Neg(refNormal) : refNormal;
  const refFaceOffset = vec2Dot(refFaceNormal, refEdge[0]);

  const contacts: ContactPoint2D[] = [];
  for (const p of clipped) {
    const sep = vec2Dot(refFaceNormal, p) - refFaceOffset;
    if (sep <= depth + 0.01) { // Tolerance: include points near the reference face boundary
      // Points behind the reference face (sep < 0) have depth = -sep.
      // Points near/on the boundary use the SAT depth, which is more accurate for
      // face-face contacts where clipped points land on the reference face edge.
      const contactDepth = sep < -1e-6 ? -sep : depth;
      contacts.push({
        position: p,
        normal: flip ? vec2Neg(normal) : normal,
        depth: contactDepth,
      });
    }
  }

  return contacts;
}

/** Find the edge of a polygon most anti-aligned with the given direction */
function findSupportEdge(vertices: Vec2[], direction: Vec2): [Vec2, Vec2] {
  const n = vertices.length;
  let bestDot = -Infinity;
  let bestIndex = 0;

  for (let i = 0; i < n; i++) {
    const d = vec2Dot(vertices[i], direction);
    if (d > bestDot) {
      bestDot = d;
      bestIndex = i;
    }
  }

  // Return the edge that includes the support vertex and is most perpendicular to direction
  const prev = (bestIndex - 1 + n) % n;
  const next = (bestIndex + 1) % n;

  const edgePrev = vec2Normalize(vec2Sub(vertices[bestIndex], vertices[prev]));
  const edgeNext = vec2Normalize(vec2Sub(vertices[next], vertices[bestIndex]));

  // Choose the edge more perpendicular to the direction
  if (Math.abs(vec2Dot(edgePrev, direction)) < Math.abs(vec2Dot(edgeNext, direction))) {
    return [vertices[prev], vertices[bestIndex]];
  } else {
    return [vertices[bestIndex], vertices[next]];
  }
}

/** Clip a segment p0-p1 against a half-plane defined by normal*x <= offset */
function clipEdge(p0: Vec2, p1: Vec2, planeNormal: Vec2, planeOffset: number): Vec2[] {
  const d0 = vec2Dot(planeNormal, p0) + planeOffset;
  const d1 = vec2Dot(planeNormal, p1) + planeOffset;

  const out: Vec2[] = [];

  if (d0 <= 0) out.push(p0);
  if (d1 <= 0) out.push(p1);

  if (d0 * d1 < 0) {
    const t = d0 / (d0 - d1);
    out.push({
      x: p0.x + t * (p1.x - p0.x),
      y: p0.y + t * (p1.y - p0.y),
    });
  }

  return out;
}

// ─── Circle-Circle Collision ────────────────────────────────────────────────

export function collideCircleCircle(bodyA: Body2D, bodyB: Body2D): ContactManifold2D | null {
  const d = vec2Sub(bodyA.position, bodyB.position);
  const dist = vec2Length(d);
  const combinedRadius = bodyA.radius + bodyB.radius;

  if (dist >= combinedRadius) return null;

  const normal = dist > 1e-10 ? vec2Scale(d, 1 / dist) : vec2(0, 1);
  const depth = combinedRadius - dist;
  const contactPoint = vec2Add(bodyB.position, vec2Scale(normal, bodyB.radius));

  return {
    bodyA: bodyA.index,
    bodyB: bodyB.index,
    normal,
    contacts: [{ position: contactPoint, normal, depth }],
  };
}

// ─── Box-Circle Collision ───────────────────────────────────────────────────

export function collideBoxCircle(box: Body2D, circle: Body2D): ContactManifold2D | null {
  // Transform circle center to box's local space
  const d = vec2Sub(circle.position, box.position);
  const c = Math.cos(-box.angle);
  const s = Math.sin(-box.angle);
  const localCenter: Vec2 = { x: c * d.x - s * d.y, y: s * d.x + c * d.y };

  // Find closest point on box to circle center
  const closest: Vec2 = {
    x: Math.max(-box.halfExtents.x, Math.min(box.halfExtents.x, localCenter.x)),
    y: Math.max(-box.halfExtents.y, Math.min(box.halfExtents.y, localCenter.y)),
  };

  const diff = vec2Sub(localCenter, closest);
  const distSq = diff.x * diff.x + diff.y * diff.y;

  if (distSq >= circle.radius * circle.radius) return null;

  const dist = Math.sqrt(distSq);

  let localNormal: Vec2;
  let contactDepth: number;

  if (dist > 1e-10) {
    localNormal = vec2Scale(diff, 1 / dist);
    contactDepth = circle.radius - dist;
  } else {
    // Circle center is inside the box
    const dx = box.halfExtents.x - Math.abs(localCenter.x);
    const dy = box.halfExtents.y - Math.abs(localCenter.y);
    if (dx < dy) {
      localNormal = localCenter.x >= 0 ? vec2(1, 0) : vec2(-1, 0);
      contactDepth = dx + circle.radius;
    } else {
      localNormal = localCenter.y >= 0 ? vec2(0, 1) : vec2(0, -1);
      contactDepth = dy + circle.radius;
    }
  }

  // Transform normal back to world space
  // localNormal points from box surface to circle center (A to B)
  // Convention: manifold normal points from B to A, so negate
  const cw = Math.cos(box.angle);
  const sw = Math.sin(box.angle);
  const outwardNormal: Vec2 = {
    x: cw * localNormal.x - sw * localNormal.y,
    y: sw * localNormal.x + cw * localNormal.y,
  };
  const manifoldNormal = vec2Neg(outwardNormal); // B to A
  const contactPoint = vec2Sub(circle.position, vec2Scale(outwardNormal, circle.radius));

  return {
    bodyA: box.index,
    bodyB: circle.index,
    normal: manifoldNormal,
    contacts: [{ position: contactPoint, normal: manifoldNormal, depth: contactDepth }],
  };
}

// ─── Dispatch ───────────────────────────────────────────────────────────────

export function collide2D(bodyA: Body2D, bodyB: Body2D): ContactManifold2D | null {
  const shapeA = bodyA.colliderShape;
  const shapeB = bodyB.colliderShape;

  if (shapeA === ColliderShapeType.Cuboid && shapeB === ColliderShapeType.Cuboid) {
    return collideBoxBox(bodyA, bodyB);
  }
  if (shapeA === ColliderShapeType.Ball && shapeB === ColliderShapeType.Ball) {
    return collideCircleCircle(bodyA, bodyB);
  }
  if (shapeA === ColliderShapeType.Cuboid && shapeB === ColliderShapeType.Ball) {
    return collideBoxCircle(bodyA, bodyB);
  }
  if (shapeA === ColliderShapeType.Ball && shapeB === ColliderShapeType.Cuboid) {
    const manifold = collideBoxCircle(bodyB, bodyA);
    if (manifold) {
      manifold.bodyA = bodyA.index;
      manifold.bodyB = bodyB.index;
      manifold.normal = vec2Neg(manifold.normal);
      for (const cc of manifold.contacts) {
        cc.normal = vec2Neg(cc.normal);
      }
    }
    return manifold;
  }
  return null;
}
