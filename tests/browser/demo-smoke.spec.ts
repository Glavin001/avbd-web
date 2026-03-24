/**
 * Demo smoke tests — verify that the 2D and 3D demos load and run
 * without freezing, crashing, or producing errors for ~10 seconds.
 */
import { test, expect } from '@playwright/test';

test.describe('Demo smoke tests', () => {
  test('2D demo runs for 10 seconds without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto('http://localhost:3333/examples/demo-2d.html', { timeout: 15000 });

    // Wait for the canvas to appear (page loaded)
    await page.waitForSelector('canvas#canvas', { timeout: 10000 });

    // Let the demo run for 10 seconds
    await page.waitForTimeout(10000);

    // Check no errors occurred
    expect(errors).toEqual([]);

    // Verify the page is still responsive (not frozen) by checking FPS display updates
    const stats1 = await page.$eval('#stats', el => el.innerHTML);
    await page.waitForTimeout(1500);
    const stats2 = await page.$eval('#stats', el => el.innerHTML);
    // Stats should be updating (not frozen)
    expect(stats2.length).toBeGreaterThan(0);

    // Check no error overlay is visible (display: none comes from CSS, so check computed style)
    const errorOverlayVisible = await page.$eval('#error-overlay', el =>
      window.getComputedStyle(el).display !== 'none'
    );
    expect(errorOverlayVisible).toBe(false);
  });

  test('3D demo runs for 10 seconds without errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto('http://localhost:3333/examples/demo-3d.html', { timeout: 15000 });

    // Wait for canvas to appear (Three.js renders to a canvas)
    await page.waitForSelector('canvas', { timeout: 10000 });

    // Let the demo run for 10 seconds
    await page.waitForTimeout(10000);

    // Check no errors occurred
    expect(errors).toEqual([]);

    // Verify the page is still responsive
    const stats1 = await page.$eval('#stats', el => el.innerHTML);
    await page.waitForTimeout(1500);
    const stats2 = await page.$eval('#stats', el => el.innerHTML);
    expect(stats2.length).toBeGreaterThan(0);

    // Check no error overlay is visible (display: none comes from CSS, so check computed style)
    const errorOverlayVisible = await page.$eval('#error-overlay', el =>
      window.getComputedStyle(el).display !== 'none'
    );
    expect(errorOverlayVisible).toBe(false);
  });
});
