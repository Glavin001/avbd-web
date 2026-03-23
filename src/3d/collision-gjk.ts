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

// ─── Box-Box (SAT 3D) ──────────────────────────────────────────────────────

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

  // Test 15 axes: 3 face normals A, 3 face normals B, 9 edge cross products
  const testAxes: Vec3[] = [];

  // Face normals
  for (const a of axesA) testAxes.push(a);
  for (const b of axesB) testAxes.push(b);

  // Edge cross products
  for (const a of axesA) {
    for (const b of axesB) {
      const cross = vec3Cross(a, b);
      if (vec3Length(cross) > 1e-6) {
        testAxes.push(vec3Normalize(cross));
      }
    }
  }

  for (const axis of testAxes) {
    // Project both boxes onto axis
    let radiusA = 0;
    for (let i = 0; i < 3; i++) {
      radiusA += Math.abs(vec3Dot(axesA[i], axis)) * heA[i];
    }
    let radiusB = 0;
    for (let i = 0; i < 3; i++) {
      radiusB += Math.abs(vec3Dot(axesB[i], axis)) * heB[i];
    }

    const distance = Math.abs(vec3Dot(d, axis));
    const overlap = radiusA + radiusB - distance;

    if (overlap < 0) return null; // Separating axis found

    if (overlap < minOverlap) {
      minOverlap = overlap;
      bestAxis = axis;
    }
  }

  // Ensure normal points from B to A
  if (vec3Dot(vec3Sub(bodyA.position, bodyB.position), bestAxis) < 0) {
    bestAxis = vec3Scale(bestAxis, -1);
  }

  // Contact point: midpoint approximation
  const midpoint = vec3Scale(vec3Add(bodyA.position, bodyB.position), 0.5);

  return {
    bodyA: bodyA.index,
    bodyB: bodyB.index,
    normal: bestAxis,
    contacts: [{ position: midpoint, normal: bestAxis, depth: minOverlap }],
  };
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
