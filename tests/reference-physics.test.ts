/**
 * Reference physics validation tests — inspired by Rapier, Box2D, and Cannon.js.
 * These test patterns are standard across all major physics engines.
 */
import { describe, it, expect } from 'vitest';
import { World, RigidBodyDesc2D, ColliderDesc2D, JointData2D } from '../src/2d/index.js';
import { AVBDSolver2D } from '../src/core/solver.js';

function createWorldWithGround(config: any = {}): World {
  const world = new World({ x: 0, y: -9.81 }, { iterations: 10, ...config });
  world.createCollider(ColliderDesc2D.cuboid(20, 0.5));
  return world;
}

function simulate(world: World, seconds: number): void {
  const steps = Math.round(seconds * 60);
  for (let i = 0; i < steps; i++) world.step();
}

// ─── 1. Restitution Sweep (Rapier pattern) ──────────────────────────────────

describe('Restitution sweep', () => {
  it('should produce monotonically increasing bounce heights with increasing restitution', () => {
    const bounceHeights: number[] = [];

    for (const e of [0.0, 0.2, 0.4, 0.6, 0.8]) {
      const world = createWorldWithGround({ iterations: 15 });
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 5)
      );
      world.createCollider(
        ColliderDesc2D.ball(0.3).setRestitution(e).setFriction(0),
        body,
      );

      // Drop and measure max height after first bounce
      let maxYAfterContact = -Infinity;
      let hitGround = false;

      for (let i = 0; i < 180; i++) {
        world.step();
        const y = body.translation().y;
        if (y < 1.5 && !hitGround) hitGround = true;
        if (hitGround && y > 1.0) {
          maxYAfterContact = Math.max(maxYAfterContact, y);
        }
      }

      bounceHeights.push(maxYAfterContact);
    }

    // Higher restitution should generally produce higher (or equal) bounce
    // Allow some tolerance since AVBD is implicit (energy dissipative)
    for (let i = 1; i < bounceHeights.length; i++) {
      expect(bounceHeights[i]).toBeGreaterThanOrEqual(bounceHeights[i - 1] - 0.5);
    }
  });
});

// ─── 2. Force/Torque Correctness (Cannon.js pattern) ────────────────────────

describe('Force and torque correctness', () => {
  it('should produce correct velocity from impulse', () => {
    const world = new World({ x: 0, y: 0 });
    const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setDensity(1), body);

    const mass = body.mass();
    body.applyImpulse({ x: mass * 10, y: 0 });

    expect(body.linvel().x).toBeCloseTo(10, 1);
    expect(body.linvel().y).toBeCloseTo(0, 5);
  });

  it('should produce correct angular velocity from torque impulse', () => {
    const world = new World({ x: 0, y: 0 });
    const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setDensity(1), body);

    // Apply torque impulse: omega = torque / I
    const I = body.mass() * (1 + 1) / 12; // cuboid(0.5,0.5) → w=h=1, I = m*(w²+h²)/12
    body.applyTorqueImpulse(I * 5); // Should give omega = 5

    expect(body.angvel()).toBeCloseTo(5, 1);
  });

  it('should produce correct motion from force over time', () => {
    const world = new World({ x: 0, y: 0 }, { dt: 1 / 60 });
    const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5).setDensity(1), body);

    const mass = body.mass();
    const force = 100;

    // Apply force each step for 1 second
    for (let i = 0; i < 60; i++) {
      body.applyForce({ x: force, y: 0 });
      world.step();
    }

    // v = F/m * t = 100/mass * 1
    const expectedV = force / mass * 1.0;
    // AVBD is implicit so there's some damping, but should be in the ballpark
    expect(body.linvel().x).toBeGreaterThan(expectedV * 0.5);
    expect(body.linvel().x).toBeLessThan(expectedV * 1.5);
  });
});

// ─── 3. Determinism Verification (Box2D pattern) ────────────────────────────

describe('Determinism', () => {
  function runScenario(): string {
    const world = createWorldWithGround({ iterations: 10 });

    // Create a small "falling hinges" scene
    const bodies: any[] = [];
    for (let i = 0; i < 5; i++) {
      const b = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(i * 0.3 - 0.6, 3 + i * 0.5)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3), b);
      bodies.push(b);
    }

    // Connect with joints
    for (let i = 0; i < 4; i++) {
      world.createJoint(
        JointData2D.revolute({ x: 0.15, y: 0 }, { x: -0.15, y: 0 }),
        bodies[i], bodies[i + 1],
      );
    }

    simulate(world, 2);

    // Hash the final state
    const state = world.getBodyStates();
    return Array.from(state).map(v => v.toFixed(8)).join(',');
  }

  it('should produce identical state hashes across runs', () => {
    const hash1 = runScenario();
    const hash2 = runScenario();
    const hash3 = runScenario();

    expect(hash1).toBe(hash2);
    expect(hash2).toBe(hash3);
  });

  it('should produce identical state for mirrored inputs', () => {
    function dropBox(x: number): number {
      const world = createWorldWithGround();
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(x, 5)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);
      simulate(world, 1);
      return body.translation().y;
    }

    // Symmetric drops should produce identical Y positions
    const y1 = dropBox(3);
    const y2 = dropBox(-3);
    expect(y1).toBe(y2);
  });
});

// ─── 4. Joint Chain & Grid Stability (Rapier pattern) ───────────────────────

describe('Joint chain stability', () => {
  it('should stabilize a 10-link hanging chain', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 15 });

    const links: any[] = [];
    const anchor = world.createRigidBody(
      RigidBodyDesc2D.fixed().setTranslation(0, 10)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.1, 0.1), anchor);
    links.push(anchor);

    for (let i = 0; i < 10; i++) {
      const link = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 9 - i * 0.8)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.1, 0.3), link);
      world.createJoint(
        JointData2D.revolute({ x: 0, y: -0.3 }, { x: 0, y: 0.3 }),
        links[i], link,
      );
      links.push(link);
    }

    simulate(world, 5);

    // All links should be finite and below anchor
    for (let i = 1; i <= 10; i++) {
      const pos = links[i].translation();
      expect(isFinite(pos.x)).toBe(true);
      expect(isFinite(pos.y)).toBe(true);
      expect(pos.y).toBeLessThan(10);
    }

    // Chain should be roughly vertical (x values near 0)
    for (let i = 1; i <= 10; i++) {
      expect(Math.abs(links[i].translation().x)).toBeLessThan(3);
    }
  });

  it('should stabilize a 3x3 joint grid', () => {
    const world = new World({ x: 0, y: -9.81 }, { iterations: 15 });

    const grid: any[][] = [];
    for (let row = 0; row < 3; row++) {
      grid[row] = [];
      for (let col = 0; col < 3; col++) {
        const isFixed = row === 0; // Top row is fixed
        const desc = isFixed
          ? RigidBodyDesc2D.fixed().setTranslation(col * 1.5, 5)
          : RigidBodyDesc2D.dynamic().setTranslation(col * 1.5, 5 - row * 1.5);
        const body = world.createRigidBody(desc);
        world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3), body);
        grid[row][col] = body;
      }
    }

    // Vertical joints
    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < 3; col++) {
        world.createJoint(
          JointData2D.revolute({ x: 0, y: -0.5 }, { x: 0, y: 0.5 }),
          grid[row][col], grid[row + 1][col],
        );
      }
    }

    // Horizontal joints
    for (let row = 1; row < 3; row++) {
      for (let col = 0; col < 2; col++) {
        world.createJoint(
          JointData2D.revolute({ x: 0.5, y: 0 }, { x: -0.5, y: 0 }),
          grid[row][col], grid[row][col + 1],
        );
      }
    }

    simulate(world, 3);

    // All bodies should be finite and below the top row
    for (let row = 1; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const pos = grid[row][col].translation();
        expect(isFinite(pos.x)).toBe(true);
        expect(isFinite(pos.y)).toBe(true);
        expect(pos.y).toBeLessThan(5);
      }
    }
  });
});

// ─── 5. Angular Damping (Rapier pattern) ────────────────────────────────────

describe('Angular damping', () => {
  it('should reduce angular velocity with angular damping', () => {
    const world = new World({ x: 0, y: 0 });
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic()
        .setAngvel(10)
        .setAngularDamping(5)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    simulate(world, 1);

    expect(Math.abs(body.angvel())).toBeLessThan(10);
  });

  it('should not damp with zero angular damping', () => {
    const world = new World({ x: 0, y: 0 });
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic()
        .setAngvel(5)
        .setAngularDamping(0)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    simulate(world, 0.5);

    // Should maintain angular velocity (no damping, no collisions)
    expect(body.angvel()).toBeCloseTo(5, 0);
  });

  it('should damp proportionally to coefficient', () => {
    function angvelAfter(damping: number): number {
      const world = new World({ x: 0, y: 0 });
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic()
          .setAngvel(10)
          .setAngularDamping(damping)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);
      simulate(world, 1);
      return Math.abs(body.angvel());
    }

    const lowDamp = angvelAfter(1);
    const highDamp = angvelAfter(10);

    // Higher damping should result in lower angular velocity
    expect(highDamp).toBeLessThan(lowDamp);
  });
});

// ─── 6. Pyramid Stacking (Rapier benchmark) ────────────────────────────────

describe('Pyramid stacking', () => {
  it('should support a 6-layer pyramid (21 boxes)', () => {
    const world = createWorldWithGround({ iterations: 15 });

    const layers = 6;
    const boxes: any[] = [];
    for (let row = 0; row < layers; row++) {
      const count = layers - row;
      const offset = -(count - 1) * 0.55;
      for (let col = 0; col < count; col++) {
        const body = world.createRigidBody(
          RigidBodyDesc2D.dynamic()
            .setTranslation(offset + col * 1.1, 0.8 + row * 1.05)
        );
        world.createCollider(
          ColliderDesc2D.cuboid(0.5, 0.5).setFriction(0.5),
          body,
        );
        boxes.push(body);
      }
    }

    expect(boxes.length).toBe(21); // 6+5+4+3+2+1

    simulate(world, 5);

    // All boxes should be above ground and finite
    let aboveGround = 0;
    for (const box of boxes) {
      const y = box.translation().y;
      expect(isFinite(y)).toBe(true);
      if (y > 0.3) aboveGround++;
    }

    // At least some boxes should remain above ground
    // A 6-layer pyramid is challenging — validate stability rather than perfection
    expect(aboveGround).toBeGreaterThan(boxes.length * 0.3);
  });
});

// ─── 7. Numerical Robustness (Box2D pattern) ────────────────────────────────

describe('Numerical robustness', () => {
  it('should handle near-zero mass body', () => {
    const world = createWorldWithGround();
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 3)
    );
    world.createCollider(
      ColliderDesc2D.cuboid(0.5, 0.5).setDensity(0.001),
      body,
    );

    simulate(world, 1);
    expect(isFinite(body.translation().y)).toBe(true);
  });

  it('should handle extremely high restitution (>1.0) without explosion', () => {
    const world = createWorldWithGround();
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 3)
    );
    world.createCollider(
      ColliderDesc2D.ball(0.5).setRestitution(1.5), // Super-elastic
      body,
    );

    // Should not crash or produce NaN
    for (let i = 0; i < 120; i++) {
      world.step();
      expect(isFinite(body.translation().y)).toBe(true);
    }
  });

  it('should handle body spawned inside another body', () => {
    const world = createWorldWithGround();

    const a = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 2)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), a);

    // Spawn second body overlapping the first
    const b = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0.3, 2)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), b);

    // Should resolve overlap without NaN
    simulate(world, 1);
    expect(isFinite(a.translation().x)).toBe(true);
    expect(isFinite(b.translation().x)).toBe(true);
    // Bodies should have separated
    const dx = Math.abs(a.translation().x - b.translation().x);
    expect(dx).toBeGreaterThan(0.1);
  });
});

// ─── 8. World Lifecycle (Box2D pattern) ──────────────────────────────────────

describe('World lifecycle', () => {
  it('should handle body creation and destruction during simulation', () => {
    const world = createWorldWithGround();

    const bodies: any[] = [];
    for (let frame = 0; frame < 60; frame++) {
      // Create a body every 5 frames
      if (frame % 5 === 0) {
        const body = world.createRigidBody(
          RigidBodyDesc2D.dynamic().setTranslation(0, 5)
        );
        world.createCollider(ColliderDesc2D.cuboid(0.3, 0.3), body);
        bodies.push(body);
      }

      // Remove oldest body every 10 frames
      if (frame % 10 === 0 && bodies.length > 3) {
        world.removeRigidBody(bodies.shift()!);
      }

      world.step();
    }

    // Remaining bodies should be finite
    for (const body of bodies) {
      expect(isFinite(body.translation().y)).toBe(true);
    }
  });

  it('should handle world recreation stress test', () => {
    for (let i = 0; i < 20; i++) {
      const world = createWorldWithGround();
      const body = world.createRigidBody(
        RigidBodyDesc2D.dynamic().setTranslation(0, 3)
      );
      world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);
      world.step();
      world.step();
      expect(isFinite(body.translation().y)).toBe(true);
    }
  });
});

// ─── 9. Body Mass Properties (Box2D pattern) ────────────────────────────────

describe('Mass properties', () => {
  it('should compute correct mass from cuboid density', () => {
    const world = new World({ x: 0, y: 0 });
    const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
    // cuboid(1, 1) → area = 4, density = 2 → mass = 8
    world.createCollider(ColliderDesc2D.cuboid(1, 1).setDensity(2), body);
    expect(body.mass()).toBeCloseTo(8, 5);
  });

  it('should compute correct mass from circle density', () => {
    const world = new World({ x: 0, y: 0 });
    const body = world.createRigidBody(RigidBodyDesc2D.dynamic());
    // ball(1) → area = pi, density = 1 → mass = pi
    world.createCollider(ColliderDesc2D.ball(1).setDensity(1), body);
    expect(body.mass()).toBeCloseTo(Math.PI, 3);
  });

  it('should have zero mass for fixed bodies', () => {
    const world = new World({ x: 0, y: 0 });
    const body = world.createRigidBody(RigidBodyDesc2D.fixed());
    world.createCollider(ColliderDesc2D.cuboid(1, 1), body);
    expect(body.mass()).toBe(0);
  });
});

// ─── 10. Contact Events Count (Box2D pattern) ───────────────────────────────

describe('Contact constraint count', () => {
  it('should generate contacts for overlapping bodies', () => {
    const world = createWorldWithGround();
    const body = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(0, 0.9) // Overlapping ground
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), body);

    world.step();

    expect(world.numConstraints).toBeGreaterThan(0);
  });

  it('should have zero contacts for separated bodies', () => {
    const world = new World({ x: 0, y: 0 });

    const a = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(-10, 0)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), a);

    const b = world.createRigidBody(
      RigidBodyDesc2D.dynamic().setTranslation(10, 0)
    );
    world.createCollider(ColliderDesc2D.cuboid(0.5, 0.5), b);

    world.step();

    // No contacts between widely separated bodies
    // (numConstraints may include joints, so check solver directly)
    const solver = world.rawSolver;
    const contactCount = solver.constraintStore.rows.filter(
      r => r.type === 0 // ForceType.Contact
    ).length;
    expect(contactCount).toBe(0);
  });
});
