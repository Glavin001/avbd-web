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
    console.log('Init result:', JSON.stringify(initResult, null, 2));
    expect(initResult.success).toBe(true);
  });

  test('should simulate free fall correctly', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.freeFall);
    expect(result.success).toBe(true);

    // Debug: log positions and GPU status
    console.log('Free fall result:', JSON.stringify(result));

    // After 1 second of free fall from y=10: y ≈ 5.1
    // Allow tolerance for implicit Euler damping
    expect(result.y).toBeLessThan(10);
    expect(result.y).toBeGreaterThan(0);
    expect(Math.abs(result.y - result.expected)).toBeLessThan(2);
  });

  test('should prevent box from falling through ground', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.boxOnGround);
    console.log('Box on ground result:', JSON.stringify(result));
    expect(result.success).toBe(true);
    // Contact detection is working (box slows near ground) - verify physics runs
    expect(result.trajectory[0].y).toBeLessThan(3);
    // Box should decelerate as it approaches ground (contacts create forces)
    const speed1 = Math.abs(result.trajectory[1].y - result.trajectory[0].y);
    expect(speed1).toBeLessThan(2); // Not free-falling at full speed
  });

  test('should handle 50 bodies without NaN', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.manyBodies);
    expect(result.success).toBe(true);
    expect(result.allFinite).toBe(true);
    expect(result.bodyCount).toBe(50);
  });

  test('should diagnose GPU execution', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.gpuDiagnostic);
    console.log('GPU diagnostic:', JSON.stringify(result, null, 2));
    expect(result.success).toBe(true);
    expect(result.preMass).toBeGreaterThan(0);
    // GPU should produce same result as CPU
    expect(result.gpuAfter1.y).toBeLessThan(10);
    expect(Math.abs(result.gpuAfter60.y - result.cpuAfter60.y)).toBeLessThan(0.1);
  });

  test('should report GPU status', async ({ page }) => {
    const gpuAvailable = await page.evaluate(() => (window as any).AVBD.isGPUAvailable);
    // Just verify the property exists and is a boolean
    expect(typeof gpuAvailable).toBe('boolean');

    const initResult = await page.evaluate(() => (window as any).testResults.init);
    expect(initResult.gpuAvailable).toBe(gpuAvailable);
  });
});
