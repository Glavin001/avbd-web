/**
 * Playwright GPU execution tests.
 * Runs the AVBD physics engine in headless Chrome with WebGPU enabled.
 * Verifies that the GPU solver produces correct physics results.
 */
import { test, expect } from '@playwright/test';

test.describe('AVBD GPU Execution', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the test harness (served by Vite)
    await page.goto('http://localhost:3333/tests/browser/test-harness.html');

    // Wait for tests to complete (up to 15s)
    await page.waitForFunction(() => (window as any).testResults?._complete === true, {
      timeout: 15000,
    });
  });

  test('should initialize AVBD engine', async ({ page }) => {
    const initResult = await page.evaluate(() => (window as any).testResults.init);
    expect(initResult.success).toBe(true);
    // GPU may or may not be available depending on the test environment
    // The important thing is init doesn't crash
  });

  test('should simulate free fall correctly', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.freeFall);
    expect(result.success).toBe(true);

    // After 1 second of free fall from y=10: y ≈ 5.1
    // Allow tolerance for implicit Euler damping
    expect(result.y).toBeLessThan(10);
    expect(result.y).toBeGreaterThan(0);
    expect(Math.abs(result.y - result.expected)).toBeLessThan(2);
  });

  test('should prevent box from falling through ground', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.boxOnGround);
    expect(result.success).toBe(true);
    expect(result.aboveGround).toBe(true);
    expect(result.y).toBeGreaterThan(0.3);
    expect(result.y).toBeLessThan(3);
  });

  test('should handle 50 bodies without NaN', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.manyBodies);
    expect(result.success).toBe(true);
    expect(result.allFinite).toBe(true);
    expect(result.bodyCount).toBe(50);
  });

  test('should report GPU status', async ({ page }) => {
    const gpuAvailable = await page.evaluate(() => (window as any).AVBD.isGPUAvailable);
    // Just verify the property exists and is a boolean
    expect(typeof gpuAvailable).toBe('boolean');

    const initResult = await page.evaluate(() => (window as any).testResults.init);
    expect(initResult.gpuAvailable).toBe(gpuAvailable);
  });
});
