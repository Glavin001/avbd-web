/**
 * Ray casting tests — based on Box2D's 14-scenario ray cast test suite.
 */
import { describe, it, expect } from 'vitest';
import { raycastClosest, raycastAll } from '../src/2d/raycast.js';
import { BodyStore2D, RigidBodyDesc2D, ColliderDesc2D } from '../src/core/rigid-body.js';

function createStore(): BodyStore2D {
  const store = new BodyStore2D();
  return store;
}

function addBox(store: BodyStore2D, x: number, y: number, hx: number, hy: number, angle: number = 0) {
  const desc = RigidBodyDesc2D.dynamic().setTranslation(x, y).setRotation(angle);
  const handle = store.addBody(desc);
  store.attachCollider(handle.index, ColliderDesc2D.cuboid(hx, hy));
  return handle;
}

function addBall(store: BodyStore2D, x: number, y: number, r: number) {
  const desc = RigidBodyDesc2D.dynamic().setTranslation(x, y);
  const handle = store.addBody(desc);
  store.attachCollider(handle.index, ColliderDesc2D.ball(r));
  return handle;
}

describe('Ray-Box intersection', () => {
  it('should hit a box directly ahead', () => {
    const store = createStore();
    addBox(store, 5, 0, 1, 1);

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).not.toBeNull();
    expect(hit!.bodyIndex).toBe(0);
    expect(hit!.point.x).toBeCloseTo(4, 1); // Left face of box at x=5-1=4
    expect(hit!.normal.x).toBeCloseTo(-1, 1); // Normal points left
  });

  it('should miss a box not in ray path', () => {
    const store = createStore();
    addBox(store, 5, 5, 1, 1); // Box above ray

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).toBeNull();
  });

  it('should hit a rotated box', () => {
    const store = createStore();
    addBox(store, 5, 0, 1, 1, Math.PI / 4); // 45-degree rotation

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).not.toBeNull();
  });

  it('should handle ray starting inside box', () => {
    const store = createStore();
    addBox(store, 0, 0, 2, 2); // Box centered at origin, ray starts inside

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    // Should hit the exit face
    expect(hit).not.toBeNull();
    expect(hit!.point.x).toBeCloseTo(2, 1);
  });

  it('should find closest box when multiple are in path', () => {
    const store = createStore();
    addBox(store, 3, 0, 0.5, 0.5); // Closer
    addBox(store, 6, 0, 0.5, 0.5); // Farther

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).not.toBeNull();
    expect(hit!.bodyIndex).toBe(0); // Should hit the closer one
  });

  it('should handle vertical ray', () => {
    const store = createStore();
    addBox(store, 0, 5, 1, 1);

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 0, y: 1 });
    expect(hit).not.toBeNull();
    expect(hit!.point.y).toBeCloseTo(4, 1);
  });

  it('should handle diagonal ray', () => {
    const store = createStore();
    addBox(store, 5, 5, 1, 1);

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 1 });
    expect(hit).not.toBeNull();
  });

  it('should handle backward ray (negative direction)', () => {
    const store = createStore();
    addBox(store, -5, 0, 1, 1);

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: -1, y: 0 });
    expect(hit).not.toBeNull();
    expect(hit!.bodyIndex).toBe(0);
  });
});

describe('Ray-Circle intersection', () => {
  it('should hit a circle directly ahead', () => {
    const store = createStore();
    addBall(store, 5, 0, 1);

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).not.toBeNull();
    expect(hit!.point.x).toBeCloseTo(4, 1);
  });

  it('should miss a circle off-axis', () => {
    const store = createStore();
    addBall(store, 5, 3, 1);

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).toBeNull();
  });

  it('should handle ray starting inside circle', () => {
    const store = createStore();
    addBall(store, 0, 0, 3);

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).not.toBeNull();
    expect(hit!.point.x).toBeCloseTo(3, 1); // Exit at radius
  });
});

describe('raycastAll', () => {
  it('should return all hits sorted by distance', () => {
    const store = createStore();
    addBox(store, 3, 0, 0.5, 0.5);
    addBox(store, 6, 0, 0.5, 0.5);
    addBox(store, 9, 0, 0.5, 0.5);

    const hits = raycastAll(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hits.length).toBe(3);
    expect(hits[0].fraction).toBeLessThan(hits[1].fraction);
    expect(hits[1].fraction).toBeLessThan(hits[2].fraction);
  });

  it('should return empty array for no hits', () => {
    const store = createStore();
    addBox(store, 5, 5, 0.5, 0.5);

    const hits = raycastAll(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hits.length).toBe(0);
  });

  it('should respect max distance', () => {
    const store = createStore();
    addBox(store, 3, 0, 0.5, 0.5);
    addBox(store, 10, 0, 0.5, 0.5); // Beyond max distance

    const hits = raycastAll(store, { x: 0, y: 0 }, { x: 1, y: 0 }, 5);
    expect(hits.length).toBe(1);
  });
});

describe('Mixed shape raycasting', () => {
  it('should find closest hit across boxes and circles', () => {
    const store = createStore();
    addBall(store, 3, 0, 0.5);  // Circle at x=3
    addBox(store, 6, 0, 0.5, 0.5); // Box at x=6

    const hit = raycastClosest(store, { x: 0, y: 0 }, { x: 1, y: 0 });
    expect(hit).not.toBeNull();
    expect(hit!.bodyIndex).toBe(0); // Circle is closer
  });
});
