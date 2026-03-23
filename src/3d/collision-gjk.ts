/**
 * 3D Collision detection for the AVBD physics engine.
 * Uses GJK for intersection testing and EPA for contact generation.
 * Also includes sphere-sphere and box-sphere specializations.
 */

import type { Vec3 } from '../core/types.js';
import type { Body3D } from '../core/rigid-body-3d.js';
import { ColliderShapeType3D } from '../core/rigid-body-3d.js';
import { vec3, vec3Add, vec3Sub, vec3Scale, vec3Dot, vec3Cross, vec3Length, vec3Normalize, quatRotateVec3 } from '../core/math.js';

export interface ContactPoint3D {
  position: Vec3;
  normal: Vec3;
  depth: number;
}

export interface ContactManifold3D {
  bodyA: number;
  bodyB: number;
  normal: Vec3;
  contacts: ContactPoint3D[];
}

// ─── AABB3D ─────────────────────────────────────────────────────────────────

export interface AABB3D {
  min: Vec3;
  max: Vec3;
}

export function aabb3DOverlap(a: AABB3D, b: AABB3D): boolean {
  return a.min.x <= b.max.x && a.max.x >= b.min.x &&
         a.min.y <= b.max.y && a.max.y >= b.min.y &&
         a.min.z <= b.max.z && a.max.z >= b.min.z;
}

export function getAABB3D(body: Body3D): AABB3D {
  if (body.colliderShape === ColliderShapeType3D.Ball) {
    return {
      min: vec3Sub(body.position, vec3(body.radius, body.radius, body.radius)),
      max: vec3Add(body.position, vec3(body.radius, body.radius, body.radius)),
    };
  }
  // For cuboid, compute AABB from rotated corners
  const he = body.halfExtents;
  const axes = [
    quatRotateVec3(body.rotation, vec3(1, 0, 0)),
    quatRotateVec3(body.rotation, vec3(0, 1, 0)),
    quatRotateVec3(body.rotation, vec3(0, 0, 1)),
  ];
  const extentX = Math.abs(axes[0].x) * he.x + Math.abs(axes[1].x) * he.y + Math.abs(axes[2].x) * he.z;
  const extentY = Math.abs(axes[0].y) * he.x + Math.abs(axes[1].y) * he.y + Math.abs(axes[2].y) * he.z;
  const extentZ = Math.abs(axes[0].z) * he.x + Math.abs(axes[1].z) * he.y + Math.abs(axes[2].z) * he.z;
  return {
    min: vec3Sub(body.position, vec3(extentX, extentY, extentZ)),
    max: vec3Add(body.position, vec3(extentX, extentY, extentZ)),
  };
}

// ─── Support functions ──────────────────────────────────────────────────────

function supportBox(body: Body3D, dir: Vec3): Vec3 {
  // Transform direction to local space
  const localDir = quatRotateVec3(
    { w: body.rotation.w, x: -body.rotation.x, y: -body.rotation.y, z: -body.rotation.z },
    dir,
  );
  // Select the vertex most aligned with the direction
  const local: Vec3 = {
    x: localDir.x >= 0 ? body.halfExtents.x : -body.halfExtents.x,
    y: localDir.y >= 0 ? body.halfExtents.y : -body.halfExtents.y,
    z: localDir.z >= 0 ? body.halfExtents.z : -body.halfExtents.z,
  };
  // Transform back to world space
  return vec3Add(body.position, quatRotateVec3(body.rotation, local));
}

function supportSphere(body: Body3D, dir: Vec3): Vec3 {
  return vec3Add(body.position, vec3Scale(vec3Normalize(dir), body.radius));
}

function support(body: Body3D, dir: Vec3): Vec3 {
  if (body.colliderShape === ColliderShapeType3D.Ball) {
    return supportSphere(body, dir);
  }
  return supportBox(body, dir);
}

// Minkowski difference support
function supportMinkowski(bodyA: Body3D, bodyB: Body3D, dir: Vec3): Vec3 {
  const a = support(bodyA, dir);
  const b = support(bodyB, vec3Scale(dir, -1));
  return vec3Sub(a, b);
}

// ─── Sphere-Sphere Collision ────────────────────────────────────────────────

export function collideSpheresSphere(bodyA: Body3D, bodyB: Body3D): ContactManifold3D | null {
  const d = vec3Sub(bodyA.position, bodyB.position);
  const dist = vec3Length(d);
  const combined = bodyA.radius + bodyB.radius;
  if (dist >= combined) return null;

  const normal = dist > 1e-10 ? vec3Scale(d, 1 / dist) : vec3(0, 1, 0);
  const depth = combined - dist;
  const contactPos = vec3Add(bodyB.position, vec3Scale(normal, bodyB.radius));

  return {
    bodyA: bodyA.index,
    bodyB: bodyB.index,
    normal,
    contacts: [{ position: contactPos, normal, depth }],
  };
}

// ─── Box-Sphere Collision ───────────────────────────────────────────────────

export function collideBoxSphere(box: Body3D, sphere: Body3D): ContactManifold3D | null {
  // Transform sphere center to box local space
  const invRot = { w: box.rotation.w, x: -box.rotation.x, y: -box.rotation.y, z: -box.rotation.z };
  const localCenter = quatRotateVec3(invRot, vec3Sub(sphere.position, box.position));

  // Clamp to box
  const closest: Vec3 = {
    x: Math.max(-box.halfExtents.x, Math.min(box.halfExtents.x, localCenter.x)),
    y: Math.max(-box.halfExtents.y, Math.min(box.halfExtents.y, localCenter.y)),
    z: Math.max(-box.halfExtents.z, Math.min(box.halfExtents.z, localCenter.z)),
  };

  const diff = vec3Sub(localCenter, closest);
  const distSq = vec3Dot(diff, diff);
  if (distSq >= sphere.radius * sphere.radius) return null;

  const dist = Math.sqrt(distSq);
  let localNormal: Vec3;
  let depth: number;

  if (dist > 1e-10) {
    localNormal = vec3Scale(diff, 1 / dist);
    depth = sphere.radius - dist;
  } else {
    const dx = box.halfExtents.x - Math.abs(localCenter.x);
    const dy = box.halfExtents.y - Math.abs(localCenter.y);
    const dz = box.halfExtents.z - Math.abs(localCenter.z);
    if (dx <= dy && dx <= dz) {
      localNormal = localCenter.x >= 0 ? vec3(1, 0, 0) : vec3(-1, 0, 0);
      depth = dx + sphere.radius;
    } else if (dy <= dz) {
      localNormal = localCenter.y >= 0 ? vec3(0, 1, 0) : vec3(0, -1, 0);
      depth = dy + sphere.radius;
    } else {
      localNormal = localCenter.z >= 0 ? vec3(0, 0, 1) : vec3(0, 0, -1);
      depth = dz + sphere.radius;
    }
  }

  // Transform back to world; negate for B-to-A convention
  const outward = quatRotateVec3(box.rotation, localNormal);
  const manifoldNormal: Vec3 = { x: -outward.x, y: -outward.y, z: -outward.z };
  const contactPoint = vec3Sub(sphere.position, vec3Scale(outward, sphere.radius));

  return {
    bodyA: box.index,
    bodyB: sphere.index,
    normal: manifoldNormal,
    contacts: [{ position: contactPoint, normal: manifoldNormal, depth }],
  };
}

// ─── Box-Box (SAT 3D) with face clipping contact generation ─────────────────

/**
 * Get the 4 vertices of a box face in world space.
 * faceAxis: which local axis the face normal is along (0=x, 1=y, 2=z)
 * sign: +1 or -1 (which side of the box)
 */
function getBoxFaceVertices(
  position: Vec3, axes: Vec3[], halfExtents: number[],
  faceAxis: number, sign: number,
): Vec3[] {
  // The two tangent axes perpendicular to the face
  const a1 = (faceAxis + 1) % 3;
  const a2 = (faceAxis + 2) % 3;
  const center = vec3Add(position, vec3Scale(axes[faceAxis], sign * halfExtents[faceAxis]));
  const t1 = vec3Scale(axes[a1], halfExtents[a1]);
  const t2 = vec3Scale(axes[a2], halfExtents[a2]);
  return [
    vec3Add(vec3Add(center, t1), t2),
    vec3Add(vec3Sub(center, t1), t2),
    vec3Sub(vec3Sub(center, t1), t2),
    vec3Sub(vec3Add(center, t1), t2),
  ];
}

/**
 * Clip a polygon (array of 3D points) against a plane defined by (planeNormal, planeDist).
 * Keeps points on the negative/zero side: dot(p, planeNormal) <= planeDist.
 */
function clipPolygonByPlane(polygon: Vec3[], planeNormal: Vec3, planeDist: number): Vec3[] {
  if (polygon.length === 0) return [];
  const out: Vec3[] = [];
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    const da = vec3Dot(a, planeNormal) - planeDist;
    const db = vec3Dot(b, planeNormal) - planeDist;
    if (da <= 0) out.push(a); // a is inside
    if ((da > 0) !== (db > 0)) {
      // edge crosses the plane — compute intersection
      const t = da / (da - db);
      out.push(vec3Add(a, vec3Scale(vec3Sub(b, a), t)));
    }
  }
  return out;
}

export function collideBoxBox3D(bodyA: Body3D, bodyB: Body3D): ContactManifold3D | null {
  // Get the 3 axes for each box
  const axesA = [
    quatRotateVec3(bodyA.rotation, vec3(1, 0, 0)),
    quatRotateVec3(bodyA.rotation, vec3(0, 1, 0)),
    quatRotateVec3(bodyA.rotation, vec3(0, 0, 1)),
  ];
  const axesB = [
    quatRotateVec3(bodyB.rotation, vec3(1, 0, 0)),
    quatRotateVec3(bodyB.rotation, vec3(0, 1, 0)),
    quatRotateVec3(bodyB.rotation, vec3(0, 0, 1)),
  ];

  const heA = [bodyA.halfExtents.x, bodyA.halfExtents.y, bodyA.halfExtents.z];
  const heB = [bodyB.halfExtents.x, bodyB.halfExtents.y, bodyB.halfExtents.z];

  const d = vec3Sub(bodyB.position, bodyA.position);

  let minOverlap = Infinity;
  let bestAxis: Vec3 = vec3(0, 1, 0);
  let bestAxisIndex = -1; // 0-2: faceA, 3-5: faceB, 6+: edge-edge

  // Test 15 axes: 3 face normals A, 3 face normals B, 9 edge cross products
  const testAxes: { axis: Vec3; type: 'faceA' | 'faceB' | 'edge'; index: number }[] = [];

  for (let i = 0; i < 3; i++) testAxes.push({ axis: axesA[i], type: 'faceA', index: i });
  for (let i = 0; i < 3; i++) testAxes.push({ axis: axesB[i], type: 'faceB', index: i });

  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const cross = vec3Cross(axesA[i], axesB[j]);
      if (vec3Length(cross) > 1e-6) {
        testAxes.push({ axis: vec3Normalize(cross), type: 'edge', index: i * 3 + j });
      }
    }
  }

  for (let i = 0; i < testAxes.length; i++) {
    const axis = testAxes[i].axis;
    let radiusA = 0;
    for (let k = 0; k < 3; k++) {
      radiusA += Math.abs(vec3Dot(axesA[k], axis)) * heA[k];
    }
    let radiusB = 0;
    for (let k = 0; k < 3; k++) {
      radiusB += Math.abs(vec3Dot(axesB[k], axis)) * heB[k];
    }

    const distance = Math.abs(vec3Dot(d, axis));
    const overlap = radiusA + radiusB - distance;

    if (overlap < 0) return null;

    if (overlap < minOverlap) {
      minOverlap = overlap;
      bestAxis = axis;
      bestAxisIndex = i;
    }
  }

  // Ensure normal points from B to A
  if (vec3Dot(vec3Sub(bodyA.position, bodyB.position), bestAxis) < 0) {
    bestAxis = vec3Scale(bestAxis, -1);
  }

  // ─── Generate multi-point contact manifold via face clipping ──────────
  const bestInfo = testAxes[bestAxisIndex];

  // Determine reference and incident faces.
  // bestAxis points from B to A. The reference face is the face closest to the other body.
  let refFaceVerts: Vec3[];
  let incFaceVerts: Vec3[];
  let refNormal: Vec3;

  if (bestInfo.type === 'faceA') {
    // Reference face on body A: face that points toward B (= anti-aligned with bestAxis)
    const dirToB = vec3Scale(bestAxis, -1); // direction from A toward B
    const sign = vec3Dot(dirToB, axesA[bestInfo.index]) > 0 ? 1 : -1;
    refNormal = vec3Scale(axesA[bestInfo.index], sign);
    refFaceVerts = getBoxFaceVertices(bodyA.position, axesA, heA, bestInfo.index, sign);
    // Incident face on B: face that points toward A (= aligned with bestAxis)
    const incAxis = findMostAntiAlignedAxis(axesB, refNormal);
    const incSign = vec3Dot(bestAxis, axesB[incAxis]) > 0 ? 1 : -1;
    incFaceVerts = getBoxFaceVertices(bodyB.position, axesB, heB, incAxis, incSign);
  } else if (bestInfo.type === 'faceB') {
    // Reference face on body B: face that points toward A (= aligned with bestAxis)
    const sign = vec3Dot(bestAxis, axesB[bestInfo.index]) > 0 ? 1 : -1;
    refNormal = vec3Scale(axesB[bestInfo.index], sign);
    refFaceVerts = getBoxFaceVertices(bodyB.position, axesB, heB, bestInfo.index, sign);
    // Incident face on A: face that points toward B (= anti-aligned with bestAxis)
    const dirToB = vec3Scale(bestAxis, -1);
    const incAxis = findMostAntiAlignedAxis(axesA, refNormal);
    const incSign = vec3Dot(dirToB, axesA[incAxis]) > 0 ? 1 : -1;
    incFaceVerts = getBoxFaceVertices(bodyA.position, axesA, heA, incAxis, incSign);
  } else {
    // Edge-edge: use midpoint fallback with single contact
    const midpoint = vec3Scale(vec3Add(bodyA.position, bodyB.position), 0.5);
    return {
      bodyA: bodyA.index,
      bodyB: bodyB.index,
      normal: bestAxis,
      contacts: [{ position: midpoint, normal: bestAxis, depth: minOverlap }],
    };
  }

  // Clip incident face against reference face's 4 side planes
  const refFaceAxis = bestInfo.type === 'faceA' ? bestInfo.index : bestInfo.index;
  let clipped = [...incFaceVerts];

  // The 4 side planes of the reference face are defined by the reference box edges
  // Each side plane has a normal along a tangent axis and offset at the box extent
  const refPos = bestInfo.type === 'faceA' ? bodyA.position : bodyB.position;
  const sideTangents = getSideTangentAxes(
    bestInfo.type === 'faceA' ? axesA : axesB,
    refFaceAxis,
    bestInfo.type === 'faceA' ? heA : heB,
    refPos,
  );

  for (const side of sideTangents) {
    clipped = clipPolygonByPlane(clipped, side.normal, side.dist);
    if (clipped.length === 0) break;
  }

  if (clipped.length === 0) {
    // Fallback: midpoint
    const midpoint = vec3Scale(vec3Add(bodyA.position, bodyB.position), 0.5);
    return {
      bodyA: bodyA.index,
      bodyB: bodyB.index,
      normal: bestAxis,
      contacts: [{ position: midpoint, normal: bestAxis, depth: minOverlap }],
    };
  }

  // Keep only points behind the reference face and compute per-point depth
  const refFaceOffset = vec3Dot(refNormal, refFaceVerts[0]);
  const contacts: ContactPoint3D[] = [];

  for (const p of clipped) {
    const sep = vec3Dot(refNormal, p) - refFaceOffset;
    if (sep <= 0.01) { // Small tolerance
      contacts.push({
        position: p,
        normal: bestAxis,
        depth: Math.max(0, -sep),
      });
    }
  }

  if (contacts.length === 0) {
    const midpoint = vec3Scale(vec3Add(bodyA.position, bodyB.position), 0.5);
    contacts.push({ position: midpoint, normal: bestAxis, depth: minOverlap });
  }

  return {
    bodyA: bodyA.index,
    bodyB: bodyB.index,
    normal: bestAxis,
    contacts,
  };
}

/** Find which axis (0, 1, 2) of the given axes is most anti-aligned with the direction */
function findMostAntiAlignedAxis(axes: Vec3[], dir: Vec3): number {
  let maxAbsDot = -1;
  let best = 0;
  for (let i = 0; i < 3; i++) {
    const absd = Math.abs(vec3Dot(axes[i], dir));
    if (absd > maxAbsDot) { maxAbsDot = absd; best = i; }
  }
  return best;
}

/**
 * Get 4 side clipping planes for a box face (perpendicular to the face).
 * Plane equation: dot(p, normal) <= dist keeps points inside the reference box.
 */
function getSideTangentAxes(
  axes: Vec3[], faceAxis: number, he: number[], refPos: Vec3,
): { normal: Vec3; dist: number }[] {
  const a1 = (faceAxis + 1) % 3;
  const a2 = (faceAxis + 2) % 3;
  // Project reference position onto each tangent axis
  const centerDotA1 = vec3Dot(refPos, axes[a1]);
  const centerDotA2 = vec3Dot(refPos, axes[a2]);
  return [
    { normal: axes[a1], dist: centerDotA1 + he[a1] },
    { normal: vec3Scale(axes[a1], -1), dist: -centerDotA1 + he[a1] },
    { normal: axes[a2], dist: centerDotA2 + he[a2] },
    { normal: vec3Scale(axes[a2], -1), dist: -centerDotA2 + he[a2] },
  ];
}

// ─── Dispatch ───────────────────────────────────────────────────────────────

export function collide3D(bodyA: Body3D, bodyB: Body3D): ContactManifold3D | null {
  const sA = bodyA.colliderShape;
  const sB = bodyB.colliderShape;

  if (sA === ColliderShapeType3D.Cuboid && sB === ColliderShapeType3D.Cuboid) {
    return collideBoxBox3D(bodyA, bodyB);
  }
  if (sA === ColliderShapeType3D.Ball && sB === ColliderShapeType3D.Ball) {
    return collideSpheresSphere(bodyA, bodyB);
  }
  if (sA === ColliderShapeType3D.Cuboid && sB === ColliderShapeType3D.Ball) {
    return collideBoxSphere(bodyA, bodyB);
  }
  if (sA === ColliderShapeType3D.Ball && sB === ColliderShapeType3D.Cuboid) {
    const m = collideBoxSphere(bodyB, bodyA);
    if (m) {
      m.bodyA = bodyA.index;
      m.bodyB = bodyB.index;
      m.normal = vec3Scale(m.normal, -1);
      for (const c of m.contacts) c.normal = vec3Scale(c.normal, -1);
    }
    return m;
  }
  return null;
}
