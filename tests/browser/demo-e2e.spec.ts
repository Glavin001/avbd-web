/**
 * Comprehensive E2E tests for AVBD demo pages.
 *
 * Tests every scene on both 2D and 3D demos with real WebGPU.
 * Detects:  JS errors, GPU device loss, error overlay visibility,
 *           silent CPU fallback, NaN positions, body escapes, explosions,
 *           and ground penetration — exactly what a human user would notice.
 */
import { test, expect, type Page } from '@playwright/test';
import { DemoPage } from './helpers/demo-page';
import { runPhysicsSanityCheck, type SanityCheckOpts } from './helpers/physics-checks';
import {
  SCENES_2D,
  SCENES_3D,
  LARGE_SCENES_2D,
  LARGE_SCENES_3D,
  WORLD_BOUNDS_2D,
  WORLD_BOUNDS_3D,
  MAX_REASONABLE_SPEED,
  GROUND_SURFACE_Y,
} from '../helpers/constants';

// ─── Core assertion helper ──────────────────────────────────────────────

/**
 * Load a demo page, select a scene, let it run, then assert everything
 * a human would expect: no errors, GPU mode, physics sanity.
 */
async function runSceneAndAssert(
  page: Page,
  dim: '2d' | '3d',
  scene: string,
  runTimeMs: number,
) {
  const demo = new DemoPage(page, dim);
  await demo.goto();
  await demo.selectScene(scene);

  // Let physics run
  await demo.runFor(runTimeMs);

  const label = `${dim} ${scene}`;

  // 1. No uncaught JS errors
  const errors = demo.getSignificantErrors();
  expect(errors, `JS errors in ${label}`).toEqual([]);

  // 2. Error overlay is hidden
  const overlayVisible = await demo.isErrorOverlayVisible();
  expect(overlayVisible, `Error overlay visible in ${label}`).toBe(false);

  // 3. Error log is empty
  const overlayText = await demo.getErrorOverlayText();
  expect(overlayText, `Error overlay text in ${label}`).toBe('');

  // 4. No step errors occurred
  const stepErrors = await demo.getStepErrors();
  expect(stepErrors, `Step errors in ${label}`).toBe(0);

  // 5. Bodies exist
  const bodyCount = await demo.getBodyCount();
  expect(bodyCount, `Body count in ${label}`).toBeGreaterThan(0);

  // 6. GPU mode is active (no silent fallback to CPU)
  const mode = await demo.getMode();
  expect(mode, `Expected GPU mode in ${label}`).toBe('GPU');

  // 7. Physics sanity — no NaN, no escapes, no explosions, no ground penetration
  const bounds = dim === '2d' ? WORLD_BOUNDS_2D : WORLD_BOUNDS_3D;
  const sanity = await page.evaluate(runPhysicsSanityCheck, {
    bounds,
    maxSpeed: MAX_REASONABLE_SPEED,
    groundY: GROUND_SURFACE_Y,
  } satisfies SanityCheckOpts);

  expect(
    sanity.ok,
    `Physics sanity failed in ${label}: ` +
      `NaN=${sanity.nanCount}, escaped=${sanity.escapedCount}, ` +
      `exploded=${sanity.explodedCount}, penetrating=${sanity.penetratingCount}`,
  ).toBe(true);

  return demo;
}

// ─── 2D Demo: All Scenes ────────────────────────────────────────────────

test.describe('2D Demo E2E', () => {
  for (const scene of SCENES_2D) {
    const isLarge = (LARGE_SCENES_2D as readonly string[]).includes(scene);
    const runTime = isLarge ? 3_000 : 5_000;

    test(`2D: ${scene}`, async ({ page }) => {
      test.setTimeout(runTime + 30_000); // run time + setup/assertion overhead
      await runSceneAndAssert(page, '2d', scene, runTime);
    });
  }
});

// ─── 3D Demo: All Scenes ────────────────────────────────────────────────

test.describe('3D Demo E2E', () => {
  for (const scene of SCENES_3D) {
    const isLarge = (LARGE_SCENES_3D as readonly string[]).includes(scene);
    const runTime = isLarge ? 3_000 : 5_000;

    test(`3D: ${scene}`, async ({ page }) => {
      test.setTimeout(runTime + 30_000);
      await runSceneAndAssert(page, '3d', scene, runTime);
    });
  }
});

// ─── Scene Switching Stress ─────────────────────────────────────────────

test.describe('Scene Switching', () => {
  test('2D: rapid scene switching does not crash', async ({ page }) => {
    const demo = new DemoPage(page, '2d');
    await demo.goto();

    for (const scene of ['stack', 'pyramid', 'friction', 'rope', 'boxRain200', 'stack']) {
      await demo.selectScene(scene);
      await demo.runFor(1_000);
    }

    expect(demo.getSignificantErrors()).toEqual([]);
    expect(await demo.isErrorOverlayVisible()).toBe(false);
    expect(await demo.getStepErrors()).toBe(0);
    expect(await demo.getMode()).toBe('GPU');
  });

  test('3D: rapid scene switching does not crash', async ({ page }) => {
    const demo = new DemoPage(page, '3d');
    await demo.goto();

    for (const scene of ['ground', 'stack', 'pyramid', 'friction', 'boxRain200', 'ground']) {
      await demo.selectScene(scene);
      await demo.runFor(1_000);
    }

    expect(demo.getSignificantErrors()).toEqual([]);
    expect(await demo.isErrorOverlayVisible()).toBe(false);
    expect(await demo.getStepErrors()).toBe(0);
    expect(await demo.getMode()).toBe('GPU');
  });
});

// ─── Long-Running Stability ─────────────────────────────────────────────

test.describe('Long-Running Stability', () => {
  test('2D stack: 15s with periodic sanity checks', async ({ page }) => {
    test.setTimeout(90_000);
    const demo = new DemoPage(page, '2d');
    await demo.goto();
    await demo.selectScene('stack');

    for (let i = 0; i < 5; i++) {
      await demo.runFor(3_000);

      expect(await demo.getBodyCount(), `bodies at check ${i}`).toBeGreaterThan(0);
      expect(await demo.isErrorOverlayVisible(), `overlay at check ${i}`).toBe(false);
      expect(await demo.getStepErrors(), `step errors at check ${i}`).toBe(0);

      const sanity = await page.evaluate(runPhysicsSanityCheck, {
        bounds: WORLD_BOUNDS_2D,
        maxSpeed: MAX_REASONABLE_SPEED,
        groundY: GROUND_SURFACE_Y,
      } satisfies SanityCheckOpts);
      expect(sanity.ok, `sanity at check ${i}: NaN=${sanity.nanCount}`).toBe(true);
    }
  });

  test('3D stack: 15s with periodic sanity checks', async ({ page }) => {
    test.setTimeout(90_000);
    const demo = new DemoPage(page, '3d');
    await demo.goto();
    await demo.selectScene('stack');

    for (let i = 0; i < 5; i++) {
      await demo.runFor(3_000);

      expect(await demo.getBodyCount(), `bodies at check ${i}`).toBeGreaterThan(0);
      expect(await demo.isErrorOverlayVisible(), `overlay at check ${i}`).toBe(false);
      expect(await demo.getStepErrors(), `step errors at check ${i}`).toBe(0);

      const sanity = await page.evaluate(runPhysicsSanityCheck, {
        bounds: WORLD_BOUNDS_3D,
        maxSpeed: MAX_REASONABLE_SPEED,
        groundY: GROUND_SURFACE_Y,
      } satisfies SanityCheckOpts);
      expect(sanity.ok, `sanity at check ${i}: NaN=${sanity.nanCount}`).toBe(true);
    }
  });
});
