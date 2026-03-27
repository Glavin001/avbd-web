/**
 * Demo E2E tests — load the actual demo pages and verify specific scenarios
 * work correctly with GPU physics (no errors, no fallback, stats updating).
 */
import { test, expect } from '@playwright/test';

const BASE = 'http://localhost:3333/examples';

// ─── Helper ────────────────────────────────────────────────────────────────

interface DemoTestOptions {
  url: string;
  canvasSelector: string;
  sceneSelector: string;
  sceneValue: string;
}

async function verifyDemoScene(
  page: import('@playwright/test').Page,
  opts: DemoTestOptions,
) {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });

  await page.goto(opts.url, { timeout: 15000 });
  await page.waitForSelector(opts.canvasSelector, { timeout: 10000 });
  await page.selectOption(opts.sceneSelector, opts.sceneValue);
  await page.waitForTimeout(3000); // let physics run

  // No JS errors (ignore ResizeObserver which is benign)
  const realErrors = errors.filter((e) => !e.includes('ResizeObserver'));
  expect(realErrors).toEqual([]);

  // No error overlay visible
  const overlayVisible = await page.$eval(
    '#error-overlay',
    (el) => getComputedStyle(el).display !== 'none',
  );
  expect(overlayVisible).toBe(false);

  // No GPU fallback or device-lost messages in error log
  const errorLog = await page.$eval('#error-log', (el) => el.textContent);
  expect(errorLog).not.toContain('falling back to CPU');
  expect(errorLog).not.toContain('GPU device lost');

  // Stats are updating (simulation is running)
  const stats = await page.$eval('#stats', (el) => el.innerHTML);
  expect(stats).toContain('Bodies');
}

// ─── 2D Demo Tests ─────────────────────────────────────────────────────────

test.describe('2D Demo E2E', () => {
  const scenes = ['stack', 'pyramid', 'friction'] as const;

  for (const scene of scenes) {
    test(`2D demo: ${scene} runs without errors`, async ({ page }) => {
      await verifyDemoScene(page, {
        url: `${BASE}/demo-2d.html`,
        canvasSelector: 'canvas#canvas',
        sceneSelector: '#sceneSelect',
        sceneValue: scene,
      });
    });
  }
});

// ─── 3D Demo Tests ─────────────────────────────────────────────────────────

test.describe('3D Demo E2E', () => {
  const scenes = ['ground', 'stack', 'pyramid', 'friction'] as const;

  for (const scene of scenes) {
    test(`3D demo: ${scene} runs without errors`, async ({ page }) => {
      await verifyDemoScene(page, {
        url: `${BASE}/demo-3d.html`,
        canvasSelector: 'canvas',
        sceneSelector: '#scene-select',
        sceneValue: scene,
      });
    });
  }
});
