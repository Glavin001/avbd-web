/**
 * Rigid body data structures for the AVBD engine.
 * Manages bodies as flat arrays (struct-of-arrays) for GPU-friendly layout.
 */

import type { Vec2, Vec3, MaterialProperties, RigidBodyHandle } from './types.js';
import type { AABB2D } from './math.js';
import { RigidBodyType, DEFAULT_MATERIAL } from './types.js';
import { vec2Rotate, vec2Add } from './math.js';

// ─── RigidBodyDesc (builder pattern like Rapier) ────────────────────────────

export class RigidBodyDesc2D {
  type: RigidBodyType = RigidBodyType.Dynamic;
  position: Vec2 = { x: 0, y: 0 };
  angle: number = 0;
  linearVelocity: Vec2 = { x: 0, y: 0 };
  angularVelocity: number = 0;
  linearDamping: number = 0;
  angularDamping: number = 0;
  mass: number = -1; // -1 = compute from collider density
  inertia: number = -1;
  gravityScale: number = 1;

  static dynamic(): RigidBodyDesc2D {
    const desc = new RigidBodyDesc2D();
    desc.type = RigidBodyType.Dynamic;
    return desc;
  }

  static fixed(): RigidBodyDesc2D {
    const desc = new RigidBodyDesc2D();
    desc.type = RigidBodyType.Fixed;
    return desc;
  }

  static kinematicPositionBased(): RigidBodyDesc2D {
    const desc = new RigidBodyDesc2D();
    desc.type = RigidBodyType.KinematicPositionBased;
    return desc;
  }

  setTranslation(x: number, y: number): RigidBodyDesc2D {
    this.position = { x, y };
    return this;
  }

  setRotation(angle: number): RigidBodyDesc2D {
    this.angle = angle;
    return this;
  }

  setLinvel(x: number, y: number): RigidBodyDesc2D {
    this.linearVelocity = { x, y };
    return this;
  }

  setAngvel(omega: number): RigidBodyDesc2D {
    this.angularVelocity = omega;
    return this;
  }

  setLinearDamping(damping: number): RigidBodyDesc2D {
    this.linearDamping = damping;
    return this;
  }

  setAngularDamping(damping: number): RigidBodyDesc2D {
    this.angularDamping = damping;
    return this;
  }

  setAdditionalMass(mass: number): RigidBodyDesc2D {
    this.mass = mass;
    return this;
  }

  setGravityScale(scale: number): RigidBodyDesc2D {
    this.gravityScale = scale;
    return this;
  }
}

// ─── ColliderDesc ───────────────────────────────────────────────────────────

export enum ColliderShapeType {
  Cuboid = 0,
  Ball = 1,
}

export class ColliderDesc2D {
  shape: ColliderShapeType;
  halfExtents: Vec2 = { x: 0, y: 0 };
  radius: number = 0;
  friction: number = DEFAULT_MATERIAL.friction;
  restitution: number = DEFAULT_MATERIAL.restitution;
  density: number = DEFAULT_MATERIAL.density;
  offset: Vec2 = { x: 0, y: 0 };

  private constructor(shape: ColliderShapeType) {
    this.shape = shape;
  }

  static cuboid(hx: number, hy: number): ColliderDesc2D {
    const desc = new ColliderDesc2D(ColliderShapeType.Cuboid);
    desc.halfExtents = { x: hx, y: hy };
    return desc;
  }

  static ball(radius: number): ColliderDesc2D {
    const desc = new ColliderDesc2D(ColliderShapeType.Ball);
    desc.radius = radius;
    return desc;
  }

  setFriction(friction: number): ColliderDesc2D {
    this.friction = friction;
    return this;
  }

  setRestitution(restitution: number): ColliderDesc2D {
    this.restitution = restitution;
    return this;
  }

  setDensity(density: number): ColliderDesc2D {
    this.density = density;
    return this;
  }

  setTranslation(x: number, y: number): ColliderDesc2D {
    this.offset = { x, y };
    return this;
  }
}

// ─── Body Store (struct-of-arrays for GPU) ──────────────────────────────────

export interface Body2D {
  index: number;
  type: RigidBodyType;
  position: Vec2;
  angle: number;
  velocity: Vec2;
  angularVelocity: number;
  mass: number;       // 0 for fixed bodies (infinite mass)
  invMass: number;
  inertia: number;    // 0 for fixed bodies
  invInertia: number;
  gravityScale: number;
  linearDamping: number;
  angularDamping: number;
  // Collider info
  colliderShape: ColliderShapeType;
  halfExtents: Vec2;
  radius: number;
  friction: number;
  restitution: number;
  // Solver state
  prevPosition: Vec2;
  prevAngle: number;
  prevVelocity: Vec2;
  inertialPosition: Vec2;
  inertialAngle: number;
  boundingRadius: number;
}

export class BodyStore2D {
  bodies: Body2D[] = [];

  addBody(desc: RigidBodyDesc2D): RigidBodyHandle {
    const index = this.bodies.length;
    const body: Body2D = {
      index,
      type: desc.type,
      position: { ...desc.position },
      angle: desc.angle,
      velocity: { ...desc.linearVelocity },
      angularVelocity: desc.angularVelocity,
      mass: 0,
      invMass: 0,
      inertia: 0,
      invInertia: 0,
      gravityScale: desc.gravityScale,
      linearDamping: desc.linearDamping,
      angularDamping: desc.angularDamping,
      colliderShape: ColliderShapeType.Cuboid,
      halfExtents: { x: 0, y: 0 },
      radius: 0,
      friction: DEFAULT_MATERIAL.friction,
      restitution: DEFAULT_MATERIAL.restitution,
      prevPosition: { ...desc.position },
      prevAngle: desc.angle,
      prevVelocity: { ...desc.linearVelocity },
      inertialPosition: { ...desc.position },
      inertialAngle: desc.angle,
      boundingRadius: 0,
    };
    this.bodies.push(body);
    return { index };
  }

  attachCollider(bodyIndex: number, desc: ColliderDesc2D): void {
    const body = this.bodies[bodyIndex];
    body.colliderShape = desc.shape;
    body.halfExtents = { ...desc.halfExtents };
    body.radius = desc.radius;
    body.friction = desc.friction;
    body.restitution = desc.restitution;

    if (body.type === RigidBodyType.Fixed) {
      body.mass = 0;
      body.invMass = 0;
      body.inertia = 0;
      body.invInertia = 0;
    } else {
      // Compute mass and inertia from shape + density
      if (desc.shape === ColliderShapeType.Cuboid) {
        const w = desc.halfExtents.x * 2;
        const h = desc.halfExtents.y * 2;
        body.mass = desc.density * w * h;
        // Box moment of inertia: (1/12) * m * (w^2 + h^2)
        body.inertia = (1 / 12) * body.mass * (w * w + h * h);
      } else {
        // Ball (circle in 2D)
        body.mass = desc.density * Math.PI * desc.radius * desc.radius;
        // Disk moment of inertia: (1/2) * m * r^2
        body.inertia = 0.5 * body.mass * desc.radius * desc.radius;
      }
      body.invMass = 1 / body.mass;
      body.invInertia = 1 / body.inertia;
    }

    // Bounding radius
    if (desc.shape === ColliderShapeType.Cuboid) {
      body.boundingRadius = Math.sqrt(
        desc.halfExtents.x * desc.halfExtents.x + desc.halfExtents.y * desc.halfExtents.y
      );
    } else {
      body.boundingRadius = desc.radius;
    }
  }

  getBody(handle: RigidBodyHandle): Body2D {
    return this.bodies[handle.index];
  }

  /** Get the 4 world-space corners of a cuboid body */
  getBoxCorners(body: Body2D): Vec2[] {
    const hx = body.halfExtents.x;
    const hy = body.halfExtents.y;
    const localCorners: Vec2[] = [
      { x: -hx, y: -hy },
      { x: hx, y: -hy },
      { x: hx, y: hy },
      { x: -hx, y: hy },
    ];
    return localCorners.map(c => vec2Add(vec2Rotate(c, body.angle), body.position));
  }

  /** Get AABB for a body */
  getAABB(body: Body2D): AABB2D {
    if (body.colliderShape === ColliderShapeType.Ball) {
      return {
        minX: body.position.x - body.radius,
        minY: body.position.y - body.radius,
        maxX: body.position.x + body.radius,
        maxY: body.position.y + body.radius,
      };
    }
    // For cuboid, compute from rotated corners
    const corners = this.getBoxCorners(body);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const c of corners) {
      if (c.x < minX) minX = c.x;
      if (c.y < minY) minY = c.y;
      if (c.x > maxX) maxX = c.x;
      if (c.y > maxY) maxY = c.y;
    }
    return { minX, minY, maxX, maxY };
  }

  get count(): number {
    return this.bodies.length;
  }
}
