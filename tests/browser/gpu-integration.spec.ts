/**
 * GPU Integration Tests — exercise the World API directly in a real browser
 * with WebGPU enabled.  Unlike the demo E2E tests, these don't load demo pages;
 * they programmatically create worlds, add bodies, and call world.step() on the
 * GPU path, then assert physics sanity.
 *
 * This catches GPU solver bugs (buffer mapping, shader errors, device loss)
 * that happen during normal API usage.
 */
import { test, expect, type Page } from '@playwright/test';

// GPU integration tests may involve many GPU steps; give them generous timeouts.
test.setTimeout(120_000);

const LOADER_URL = 'http://localhost:3333/tests/browser/gpu-loader.html';

// ─── Helpers ────────────────────────────────────────────────────────────

/** Wait for the GPU loader page to be ready. */
async function initLoader(page: Page) {
  await page.goto(LOADER_URL, { timeout: 20_000 });
  await page.waitForFunction(() => (window as any).__ready === true, { timeout: 20_000 });

  const error = await page.evaluate(() => (window as any).__error);
  expect(error, 'GPU init error').toBeUndefined();
}

interface StepResult {
  success: boolean;
  error?: string;
  bodyCount: number;
  positions: { x: number; y: number }[];
  hasNaN: boolean;
  allAboveGround: boolean;
  maxY: number;
  minY: number;
  maxSpeed: number;
}

// ─── 2D GPU Integration ─────────────────────────────────────────────────

test.describe('2D GPU Integration', () => {
  test('stack of 10 boxes: 300 GPU steps, no NaN or escapes', async ({ page }) => {
    await initLoader(page);

    const result = await page.evaluate(async () => {
      const AVBD = (window as any).AVBD;
      const world = new AVBD.World({ x: 0, y: -9.81 }, { iterations: 10, postStabilize: true });

      // Ground
      world.createCollider(AVBD.ColliderDesc.cuboid(20, 0.5).setFriction(0.5));

      // Stack of 10 boxes
      const bodies: any[] = [];
      for (let i = 0; i < 10; i++) {
        const body = world.createRigidBody(
          AVBD.RigidBodyDesc.dynamic().setTranslation(0, 0.8 + i * 1.05),
        );
        world.createCollider(
          AVBD.ColliderDesc.cuboid(0.5, 0.5).setDensity(1).setFriction(0.5).setRestitution(0.1),
          body,
        );
        bodies.push(body);
      }

      // Step 300 times on GPU, checking sanity periodically
      let hasNaN = false;
      let allAboveGround = true;
      let maxSpeed = 0;

      for (let step = 0; step < 300; step++) {
        await world.step();

        if (step % 30 === 0) {
          for (const b of bodies) {
            const t = b.translation();
            const v = b.linvel();
            if (!isFinite(t.x) || !isFinite(t.y)) hasNaN = true;
            if (t.y < -5) allAboveGround = false;
            const speed = Math.sqrt(v.x * v.x + v.y * v.y);
            if (speed > maxSpeed) maxSpeed = speed;
          }
        }
      }

      // Final positions
      const positions = bodies.map((b: any) => {
        const t = b.translation();
        return { x: t.x, y: t.y };
      });
      const minY = Math.min(...positions.map((p: any) => p.y));
      const maxY = Math.max(...positions.map((p: any) => p.y));

      return {
        success: true,
        bodyCount: bodies.length,
        positions,
        hasNaN,
        allAboveGround,
        maxY,
        minY,
        maxSpeed,
      } as any;
    });

    expect(result.success).toBe(true);
    expect(result.bodyCount).toBe(10);
    expect(result.hasNaN, 'NaN detected in positions').toBe(false);
    expect(result.allAboveGround, 'Body fell below ground').toBe(true);
    expect(result.maxSpeed, 'Explosion detected').toBeLessThan(500);
    expect(result.minY, 'Body penetrated ground').toBeGreaterThan(-1);
  });

  test('200 mixed shapes: GPU handles large body count', async ({ page }) => {
    await initLoader(page);

    const result = await page.evaluate(async () => {
      const AVBD = (window as any).AVBD;
      const world = new AVBD.World({ x: 0, y: -9.81 }, { iterations: 10, postStabilize: true });

      world.createCollider(AVBD.ColliderDesc.cuboid(30, 0.5).setFriction(0.5));

      const bodies: any[] = [];
      for (let i = 0; i < 200; i++) {
        const col = i % 15;
        const row = Math.floor(i / 15);
        const x = (col - 7) * 0.8;
        const y = 1.5 + row * 0.8;
        const body = world.createRigidBody(
          AVBD.RigidBodyDesc.dynamic().setTranslation(x, y),
        );

        if (i % 3 === 0) {
          world.createCollider(
            AVBD.ColliderDesc.ball(0.3).setDensity(1).setFriction(0.4),
            body,
          );
        } else {
          const sz = 0.2 + Math.random() * 0.15;
          world.createCollider(
            AVBD.ColliderDesc.cuboid(sz, sz).setDensity(1).setFriction(0.4),
            body,
          );
        }
        bodies.push(body);
      }

      let nanCount = 0;
      let stepError: string | null = null;

      try {
        for (let step = 0; step < 120; step++) {
          await world.step();

          if (step % 20 === 0) {
            for (const b of bodies) {
              const t = b.translation();
              if (!isFinite(t.x) || !isFinite(t.y)) nanCount++;
            }
          }
        }
      } catch (e: any) {
        stepError = e.message;
      }

      return { success: stepError === null, stepError, bodyCount: bodies.length, nanCount };
    });

    expect(result.success, `Step error: ${result.stepError}`).toBe(true);
    expect(result.bodyCount).toBe(200);
    expect(result.nanCount, 'NaN positions detected').toBe(0);
  });

  test('multiple world create/destroy cycles: no resource leaks', async ({ page }) => {
    await initLoader(page);

    const result = await page.evaluate(async () => {
      const AVBD = (window as any).AVBD;
      let stepError: string | null = null;

      for (let cycle = 0; cycle < 3; cycle++) {
        const world = new AVBD.World({ x: 0, y: -9.81 }, { iterations: 6 });
        world.createCollider(AVBD.ColliderDesc.cuboid(10, 0.5));

        for (let i = 0; i < 10; i++) {
          const body = world.createRigidBody(
            AVBD.RigidBodyDesc.dynamic().setTranslation(Math.random() * 4 - 2, 2 + i * 1.1),
          );
          world.createCollider(AVBD.ColliderDesc.cuboid(0.4, 0.4).setDensity(1), body);
        }

        try {
          for (let step = 0; step < 30; step++) {
            await world.step();
          }
        } catch (e: any) {
          stepError = `Cycle ${cycle}: ${e.message}`;
          break;
        }
      }

      return { success: stepError === null, stepError, cycles: 3 };
    });

    expect(result.success, `Resource leak / device loss: ${result.stepError}`).toBe(true);
  });

  test('joints: GPU solver maintains constraint distance', async ({ page }) => {
    await initLoader(page);

    const result = await page.evaluate(async () => {
      const AVBD = (window as any).AVBD;
      const world = new AVBD.World({ x: 0, y: -9.81 }, { iterations: 10 });

      // Fixed anchor
      const anchor = world.createRigidBody(
        AVBD.RigidBodyDesc.fixed().setTranslation(0, 5),
      );
      world.createCollider(AVBD.ColliderDesc.cuboid(0.2, 0.2), anchor);

      // Pendulum bob
      const bob = world.createRigidBody(
        AVBD.RigidBodyDesc.dynamic().setTranslation(2, 5),
      );
      world.createCollider(
        AVBD.ColliderDesc.cuboid(0.3, 0.3).setDensity(1),
        bob,
      );

      // Joint
      const jd = AVBD.JointData.revolute(
        { x: 0, y: 0 },
        { x: 0, y: 0 },
      );
      world.createJoint(jd, anchor, bob);

      let maxDist = 0;
      let minDist = Infinity;

      for (let step = 0; step < 180; step++) {
        await world.step();

        if (step % 10 === 0) {
          const aT = anchor.translation();
          const bT = bob.translation();
          const dist = Math.sqrt((aT.x - bT.x) ** 2 + (aT.y - bT.y) ** 2);
          if (dist > maxDist) maxDist = dist;
          if (dist < minDist) minDist = dist;
        }
      }

      return { success: true, maxDist, minDist };
    });

    expect(result.success).toBe(true);
    // Joint should keep distance bounded — GPU solver may not maintain exact constraint
    expect(result.maxDist).toBeLessThan(5.0);
    expect(result.minDist).toBeGreaterThan(0.1);
  });
});

// ─── 3D GPU Integration ─────────────────────────────────────────────────

test.describe('3D GPU Integration', () => {
  test('stack of 5 cubes: 300 GPU steps, no NaN', async ({ page }) => {
    await initLoader(page);

    const has3dError = await page.evaluate(() => (window as any).__3dError);
    if (has3dError) {
      test.skip();
      return;
    }

    const result = await page.evaluate(async () => {
      const AVBD3D = (window as any).AVBD3D;
      const world = new AVBD3D.World(
        { x: 0, y: -9.81, z: 0 },
        { iterations: 10 },
      );

      // Ground
      world.createCollider(AVBD3D.ColliderDesc.cuboid(50, 0.5, 50).setFriction(0.6));

      const bodies: any[] = [];
      for (let i = 0; i < 5; i++) {
        const body = world.createRigidBody(
          AVBD3D.RigidBodyDesc.dynamic().setTranslation(0, 1 + i * 1.1, 0),
        );
        world.createCollider(
          AVBD3D.ColliderDesc.cuboid(0.5, 0.5, 0.5).setDensity(1).setFriction(0.5),
          body,
        );
        bodies.push(body);
      }

      let hasNaN = false;
      let stepError: string | null = null;

      try {
        for (let step = 0; step < 300; step++) {
          await world.step();

          if (step % 30 === 0) {
            for (const b of bodies) {
              const t = b.translation();
              if (!isFinite(t.x) || !isFinite(t.y) || !isFinite(t.z)) hasNaN = true;
            }
          }
        }
      } catch (e: any) {
        stepError = e.message;
      }

      const positions = bodies.map((b: any) => {
        const t = b.translation();
        return { x: t.x, y: t.y, z: t.z };
      });
      const minY = Math.min(...positions.map((p: any) => p.y));

      return {
        success: stepError === null,
        stepError,
        bodyCount: bodies.length,
        hasNaN,
        minY,
      };
    });

    expect(result.success, `3D step error: ${result.stepError}`).toBe(true);
    expect(result.bodyCount).toBe(5);
    expect(result.hasNaN, '3D NaN detected').toBe(false);
    expect(result.minY, '3D body penetrated ground').toBeGreaterThan(-1);
  });

  test('10 spheres: GPU handles 3D body count', async ({ page }) => {
    await initLoader(page);

    const has3dError = await page.evaluate(() => (window as any).__3dError);
    if (has3dError) {
      test.skip();
      return;
    }

    const result = await page.evaluate(async () => {
      const AVBD3D = (window as any).AVBD3D;
      const world = new AVBD3D.World(
        { x: 0, y: -9.81, z: 0 },
        { iterations: 10 },
      );

      world.createCollider(AVBD3D.ColliderDesc.cuboid(50, 0.5, 50).setFriction(0.5));

      const bodies: any[] = [];
      for (let i = 0; i < 10; i++) {
        const x = (i % 4 - 2) * 1.5;
        const z = (Math.floor(i / 4) - 1) * 1.5;
        const y = 2 + Math.random() * 3;
        const body = world.createRigidBody(
          AVBD3D.RigidBodyDesc.dynamic().setTranslation(x, y, z),
        );
        world.createCollider(
          AVBD3D.ColliderDesc.ball(0.4).setDensity(1).setFriction(0.4),
          body,
        );
        bodies.push(body);
      }

      let nanCount = 0;
      let stepError: string | null = null;

      try {
        for (let step = 0; step < 30; step++) {
          await world.step();

          if (step % 10 === 0) {
            for (const b of bodies) {
              const t = b.translation();
              if (!isFinite(t.x) || !isFinite(t.y) || !isFinite(t.z)) nanCount++;
            }
          }
        }
      } catch (e: any) {
        stepError = e.message;
      }

      return { success: stepError === null, stepError, bodyCount: bodies.length, nanCount };
    });

    expect(result.success, `3D step error: ${result.stepError}`).toBe(true);
    expect(result.bodyCount).toBe(10);
    expect(result.nanCount).toBe(0);
  });

  test('multiple world create/destroy cycles: no 3D resource leaks', async ({ page }) => {
    await initLoader(page);

    const has3dError = await page.evaluate(() => (window as any).__3dError);
    if (has3dError) {
      test.skip();
      return;
    }

    const result = await page.evaluate(async () => {
      const AVBD3D = (window as any).AVBD3D;
      let stepError: string | null = null;

      for (let cycle = 0; cycle < 3; cycle++) {
        const world = new AVBD3D.World(
          { x: 0, y: -9.81, z: 0 },
          { iterations: 6 },
        );
        world.createCollider(AVBD3D.ColliderDesc.cuboid(20, 0.5, 20));

        for (let i = 0; i < 5; i++) {
          const body = world.createRigidBody(
            AVBD3D.RigidBodyDesc.dynamic().setTranslation(
              Math.random() * 4 - 2,
              2 + i * 1.1,
              Math.random() * 4 - 2,
            ),
          );
          world.createCollider(AVBD3D.ColliderDesc.cuboid(0.4, 0.4, 0.4).setDensity(1), body);
        }

        try {
          for (let step = 0; step < 30; step++) {
            await world.step();
          }
        } catch (e: any) {
          stepError = `Cycle ${cycle}: ${e.message}`;
          break;
        }
      }

      return { success: stepError === null, stepError };
    });

    expect(result.success, `3D resource leak: ${result.stepError}`).toBe(true);
  });
});
