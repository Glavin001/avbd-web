/**
 * 3D Rigid body data structures for the AVBD engine.
 * 6-DOF: position (x,y,z) + orientation (quaternion w,x,y,z).
 */

import type { Vec3, Quat, RigidBodyHandle } from './types.js';
import { RigidBodyType, DEFAULT_MATERIAL } from './types.js';
import { vec3Add, vec3Sub, vec3Scale, vec3Cross, vec3Length, quatIdentity, quatRotateVec3, quatFromAxisAngle, quatMul, quatNormalize } from './math.js';

// ─── RigidBodyDesc3D (builder pattern) ──────────────────────────────────────

export class RigidBodyDesc3D {
  type: RigidBodyType = RigidBodyType.Dynamic;
  position: Vec3 = { x: 0, y: 0, z: 0 };
  rotation: Quat = quatIdentity();
  linearVelocity: Vec3 = { x: 0, y: 0, z: 0 };
  angularVelocity: Vec3 = { x: 0, y: 0, z: 0 };
  linearDamping: number = 0;
  angularDamping: number = 0;
  gravityScale: number = 1;

  static dynamic(): RigidBodyDesc3D {
    const desc = new RigidBodyDesc3D();
    desc.type = RigidBodyType.Dynamic;
    return desc;
  }

  static fixed(): RigidBodyDesc3D {
    const desc = new RigidBodyDesc3D();
    desc.type = RigidBodyType.Fixed;
    return desc;
  }

  static kinematicPositionBased(): RigidBodyDesc3D {
    const desc = new RigidBodyDesc3D();
    desc.type = RigidBodyType.KinematicPositionBased;
    return desc;
  }

  setTranslation(x: number, y: number, z: number): RigidBodyDesc3D {
    this.position = { x, y, z };
    return this;
  }

  setRotation(q: Quat): RigidBodyDesc3D {
    this.rotation = { ...q };
    return this;
  }

  setLinvel(x: number, y: number, z: number): RigidBodyDesc3D {
    this.linearVelocity = { x, y, z };
    return this;
  }

  setAngvel(x: number, y: number, z: number): RigidBodyDesc3D {
    this.angularVelocity = { x, y, z };
    return this;
  }

  setLinearDamping(damping: number): RigidBodyDesc3D {
    this.linearDamping = damping;
    return this;
  }

  setAngularDamping(damping: number): RigidBodyDesc3D {
    this.angularDamping = damping;
    return this;
  }

  setGravityScale(scale: number): RigidBodyDesc3D {
    this.gravityScale = scale;
    return this;
  }
}

// ─── ColliderDesc3D ─────────────────────────────────────────────────────────

export enum ColliderShapeType3D {
  Cuboid = 0,
  Ball = 1,
}

export class ColliderDesc3D {
  shape: ColliderShapeType3D;
  halfExtents: Vec3 = { x: 0, y: 0, z: 0 };
  radius: number = 0;
  friction: number = DEFAULT_MATERIAL.friction;
  restitution: number = DEFAULT_MATERIAL.restitution;
  density: number = DEFAULT_MATERIAL.density;

  private constructor(shape: ColliderShapeType3D) {
    this.shape = shape;
  }

  static cuboid(hx: number, hy: number, hz: number): ColliderDesc3D {
    const desc = new ColliderDesc3D(ColliderShapeType3D.Cuboid);
    desc.halfExtents = { x: hx, y: hy, z: hz };
    return desc;
  }

  static ball(radius: number): ColliderDesc3D {
    const desc = new ColliderDesc3D(ColliderShapeType3D.Ball);
    desc.radius = radius;
    return desc;
  }

  setFriction(f: number): ColliderDesc3D { this.friction = f; return this; }
  setRestitution(r: number): ColliderDesc3D { this.restitution = r; return this; }
  setDensity(d: number): ColliderDesc3D { this.density = d; return this; }
}

// ─── Body3D ─────────────────────────────────────────────────────────────────

export interface Body3D {
  index: number;
  type: RigidBodyType;
  position: Vec3;
  rotation: Quat;
  velocity: Vec3;
  angularVelocity: Vec3;
  mass: number;
  invMass: number;
  /** Diagonal inertia tensor (principal axes) */
  inertia: Vec3;
  invInertia: Vec3;
  gravityScale: number;
  linearDamping: number;
  angularDamping: number;
  colliderShape: ColliderShapeType3D;
  halfExtents: Vec3;
  radius: number;
  friction: number;
  restitution: number;
  // Solver state
  prevPosition: Vec3;
  prevRotation: Quat;
  prevVelocity: Vec3;
  inertialPosition: Vec3;
  inertialRotation: Quat;
  boundingRadius: number;
}

export class BodyStore3D {
  bodies: Body3D[] = [];

  addBody(desc: RigidBodyDesc3D): RigidBodyHandle {
    const index = this.bodies.length;
    const body: Body3D = {
      index,
      type: desc.type,
      position: { ...desc.position },
      rotation: { ...desc.rotation },
      velocity: { ...desc.linearVelocity },
      angularVelocity: { ...desc.angularVelocity },
      mass: 0, invMass: 0,
      inertia: { x: 0, y: 0, z: 0 },
      invInertia: { x: 0, y: 0, z: 0 },
      gravityScale: desc.gravityScale,
      linearDamping: desc.linearDamping,
      angularDamping: desc.angularDamping,
      colliderShape: ColliderShapeType3D.Cuboid,
      halfExtents: { x: 0, y: 0, z: 0 },
      radius: 0,
      friction: DEFAULT_MATERIAL.friction,
      restitution: DEFAULT_MATERIAL.restitution,
      prevPosition: { ...desc.position },
      prevRotation: { ...desc.rotation },
      prevVelocity: { ...desc.linearVelocity },
      inertialPosition: { ...desc.position },
      inertialRotation: { ...desc.rotation },
      boundingRadius: 0,
    };
    this.bodies.push(body);
    return { index };
  }

  attachCollider(bodyIndex: number, desc: ColliderDesc3D): void {
    const body = this.bodies[bodyIndex];
    body.colliderShape = desc.shape;
    body.halfExtents = { ...desc.halfExtents };
    body.radius = desc.radius;
    body.friction = desc.friction;
    body.restitution = desc.restitution;

    if (body.type === RigidBodyType.Fixed) {
      body.mass = 0; body.invMass = 0;
      body.inertia = { x: 0, y: 0, z: 0 };
      body.invInertia = { x: 0, y: 0, z: 0 };
    } else {
      if (desc.shape === ColliderShapeType3D.Cuboid) {
        const w = desc.halfExtents.x * 2;
        const h = desc.halfExtents.y * 2;
        const d = desc.halfExtents.z * 2;
        body.mass = desc.density * w * h * d;
        // Box inertia tensor (diagonal, principal axes)
        body.inertia = {
          x: (1 / 12) * body.mass * (h * h + d * d),
          y: (1 / 12) * body.mass * (w * w + d * d),
          z: (1 / 12) * body.mass * (w * w + h * h),
        };
      } else {
        // Sphere
        body.mass = desc.density * (4 / 3) * Math.PI * desc.radius ** 3;
        const I = (2 / 5) * body.mass * desc.radius ** 2;
        body.inertia = { x: I, y: I, z: I };
      }
      body.invMass = 1 / body.mass;
      body.invInertia = {
        x: 1 / body.inertia.x,
        y: 1 / body.inertia.y,
        z: 1 / body.inertia.z,
      };
    }

    if (desc.shape === ColliderShapeType3D.Cuboid) {
      body.boundingRadius = vec3Length(desc.halfExtents);
    } else {
      body.boundingRadius = desc.radius;
    }
  }

  getBody(handle: RigidBodyHandle): Body3D {
    return this.bodies[handle.index];
  }

  get count(): number { return this.bodies.length; }
}
