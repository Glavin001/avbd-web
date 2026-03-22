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
    console.log('Free fall result:', JSON.stringify(result));

    // After 60 steps (1 second) from y=10: y ≈ 10 - 0.5*9.81*1² ≈ 5.1
    // GPU uses f32, so allow small tolerance vs analytical
    expect(result.y).toBeGreaterThan(4.0);
    expect(result.y).toBeLessThan(6.5);
    expect(Math.abs(result.y - result.expected)).toBeLessThan(0.5);

    // Verify monotonic descent during free fall
    const positions = result.positions;
    for (let i = 1; i < positions.length; i++) {
      expect(positions[i].y).toBeLessThan(positions[i - 1].y);
    }
  });

  test('should prevent box from falling through ground', async ({ page }) => {
    const result = await page.evaluate(() => (window as any).testResults.boxOnGround);
    console.log('Box on ground result:', JSON.stringify(result));
    expect(result.success).toBe(true);

    // Ground top at y=0.5, box half-height=0.5, so resting position ≈ y=1.0
    // Box must stay above ground at all times
    expect(result.aboveGround).toBe(true);
    expect(result.y).toBeGreaterThan(0.5);  // Well above ground surface
    expect(result.y).toBeLessThan(3.0);     // Settled, not still falling or bouncing high

    // Verify box approaches ground and settles (not free-falling through)
    // Find the minimum y in the trajectory (should be above 0.5)
    const minY = Math.min(...result.trajectory.map((t: any) => t.y));
    expect(minY).toBeGreaterThan(0.5);  // Never penetrates ground

    // Final position should be near resting height y≈1.0
    expect(result.y).toBeGreaterThan(0.7);
    expect(result.y).toBeLessThan(2.0);

    // GPU and CPU should both keep box above ground
    expect(result.cpuY).toBeGreaterThan(0.5);
    expect(result.cpuY).toBeLessThan(3.0);
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
    expect(result.isGPU).toBe(true);

    // First step: GPU and CPU should nearly match
    expect(Math.abs(result.gpuAfter1.y - result.cpuAfter1.y)).toBeLessThan(0.001);

    // After 60 steps of free fall: GPU (f32) and CPU (f64) diverge slightly
    expect(result.gpuAfter60.y).toBeLessThan(10);
    expect(result.gpuAfter60.y).toBeGreaterThan(0);
    expect(Math.abs(result.gpuAfter60.y - result.cpuAfter60.y)).toBeLessThan(0.05);
  });

  test('should report GPU status', async ({ page }) => {
    const gpuAvailable = await page.evaluate(() => (window as any).AVBD.isGPUAvailable);
    // Just verify the property exists and is a boolean
    expect(typeof gpuAvailable).toBe('boolean');

    const initResult = await page.evaluate(() => (window as any).testResults.init);
    expect(initResult.gpuAvailable).toBe(gpuAvailable);
  });
});
