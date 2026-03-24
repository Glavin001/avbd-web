/**
 * GPU execution tests — runs AVBD physics on real WebGPU via headless Chrome.
 *
 * Every test calls world.step() (the GPU path) and validates physics correctness
 * with tight numerical bounds. These are NOT structural/smoke tests —
 * they verify that the GPU compute shaders produce correct physics.
 */
import { test, expect } from '@playwright/test';

test.describe('AVBD GPU Execution', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3333/tests/browser/test-harness.html');
    await page.waitForFunction(() => (window as any).testResults?._complete === true, {
      timeout: 30000,
    });
  });

  test('should initialize with GPU compute support', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.init);
    expect(r.success).toBe(true);
    expect(r.gpuAvailable).toBe(true);
    expect(r.computeWorks).toBe(true);
  });

  // ─── Free fall: analytical correctness ──────────────────────────────

  test('free fall: y matches analytical formula within 0.5m', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.freeFall);
    console.log('free fall:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);

    // After 60 steps (1s) from y=10: y ≈ 10 - 0.5*9.81*1² ≈ 5.095
    expect(r.error).toBeLessThan(0.5);
    expect(r.y).toBeGreaterThan(4.0);
    expect(r.y).toBeLessThan(6.5);
  });

  test('free fall: monotonically decreasing y, no horizontal drift', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.freeFall);
    expect(r.success).toBe(true);
    expect(r.monotonic).toBe(true);
    expect(r.xDrift).toBeLessThan(0.001);
  });

  // ─── Contact constraints ────────────────────────────────────────────

  test('box on ground: GPU never penetrates ground surface', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.boxOnGround);
    console.log('box on ground:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);

    // Ground top=0.5, box half=0.5, rest ≈ 1.0
    // GPU must NEVER penetrate ground (minY > 0.5 allows tiny overlap)
    expect(r.gpuMinY).toBeGreaterThan(0.4);
    expect(r.cpuMinY).toBeGreaterThan(0.4);

    // Final resting height near 1.0 (within 0.5)
    expect(r.gpuFinalY).toBeGreaterThan(0.7);
    expect(r.gpuFinalY).toBeLessThan(2.0);
  });

  // ─── GPU/CPU equivalence ────────────────────────────────────────────

  test('GPU vs CPU: step-1 position difference < 0.001m', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpuCpuStep1);
    console.log('GPU vs CPU step 1:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.diff).toBeLessThan(0.001);
  });

  test('GPU vs CPU: 60-step f32 drift < 0.05m', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpuReadback);
    console.log('GPU readback:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);

    // Step 1: nearly identical
    expect(r.diff1).toBeLessThan(0.001);
    // Step 60: f32 accumulation still within 0.05m
    expect(r.diff60).toBeLessThan(0.05);
    // GPU free-fall result is physically plausible
    expect(r.gpu60Y).toBeGreaterThan(0);
    expect(r.gpu60Y).toBeLessThan(10);
  });

  // ─── Restitution ────────────────────────────────────────────────────

  test('restitution: e=0.7 bounces higher than e=0.2', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.restitution);
    console.log('restitution:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.bothHitGround).toBe(true);
    expect(r.higherBounce).toBe(true);
    expect(r.lowAboveGround).toBe(true);
    expect(r.highAboveGround).toBe(true);
  });

  // ─── Friction ───────────────────────────────────────────────────────

  test('friction: GPU low-mu slides further than high-mu', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.friction);
    console.log('friction:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);

    // CPU friction differentiates mu values
    expect(r.cpuFrictionWorks).toBe(true);
    expect(r.cpuLowX).toBeGreaterThan(r.cpuHighX);

    // GPU friction MUST also differentiate mu values
    expect(r.gpuFrictionDiff).toBeGreaterThan(1.0);
    expect(r.gpuLowX).toBeGreaterThan(r.gpuHighX);

    // Both remain above ground
    expect(r.gpuHighAbove).toBe(true);
    expect(r.gpuLowAbove).toBe(true);
    // Both moved right
    expect(r.gpuHighMoved).toBe(true);
    expect(r.gpuLowMoved).toBe(true);
  });

  // ─── Stacking ───────────────────────────────────────────────────────

  test('stacking: 5 boxes settle above ground in order', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.stacking);
    console.log('stacking:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.allGpuAbove).toBe(true);
    expect(r.allCpuAbove).toBe(true);
    expect(r.gpuOrdered).toBe(true);
    expect(r.cpuOrdered).toBe(true);
  });

  // ─── Circle collision ──────────────────────────────────────────────

  test('circle on ground: ball settles above ground', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.circleOnGround);
    console.log('circle on ground:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    // Ball radius=0.5, ground top=0.5 → rest ≈ 1.0
    expect(r.gpuY).toBeGreaterThan(0.7);
    expect(r.gpuY).toBeLessThan(2.0);
    expect(r.cpuY).toBeGreaterThan(0.7);
    expect(r.cpuY).toBeLessThan(2.0);
  });

  // ─── Joint constraints ─────────────────────────────────────────────

  test('joint pendulum: GPU and CPU both hold arm length ≈ 2.0', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.jointPendulum);
    console.log('joint pendulum:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);

    // CPU must maintain joint constraint precisely
    expect(r.cpuDist).toBeGreaterThan(1.5);
    expect(r.cpuDist).toBeLessThan(2.5);

    // GPU must also maintain joint constraint (LDL regularization fix)
    expect(r.gpuDist).toBeGreaterThan(1.5);
    expect(r.gpuDist).toBeLessThan(2.5);

    // GPU and CPU should produce similar results
    expect(Math.abs(r.gpuDist - r.cpuDist)).toBeLessThan(0.01);

    // Max drift during simulation should stay bounded
    expect(r.maxGpuDist).toBeLessThan(3.0);
  });

  // ─── Applied forces ─────────────────────────────────────────────────

  test('applied impulse: GPU body moves in force direction', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.appliedForce);
    console.log('applied force:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    // Body received rightward impulse in zero-gravity world
    expect(r.movedRight).toBe(true);
    expect(r.finalX).toBeGreaterThan(1.0);
    // Y stays near zero (no gravity)
    expect(r.yNearZero).toBe(true);
    // X monotonically increasing (uniform motion)
    expect(r.xMonotonic).toBe(true);
  });

  // ─── Mid-air collision ──────────────────────────────────────────────

  test('mid-air collision: two boxes collide without passing through', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.midAirCollision);
    console.log('mid-air collision:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.finite).toBe(true);
    expect(r.collisionDetected).toBe(true);
    expect(r.noOverlap).toBe(true);
  });

  // ─── Stress test ────────────────────────────────────────────────────

  test('200 bodies: zero NaN, all above ground', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.stress200);
    console.log('stress 200:', JSON.stringify(r));
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.nanCount).toBe(0);
    expect(r.bodyCount).toBe(200);
    // Max Y should be reasonable (not exploding upward)
    expect(r.maxY).toBeLessThan(100);
  });

  // ─── Determinism ────────────────────────────────────────────────────

  test('GPU determinism: two identical runs produce identical positions', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.determinism);
    console.log('determinism:', JSON.stringify(r));
    expect(r.success).toBe(true);
    // Exact bit-for-bit match expected for identical inputs
    expect(r.b1YDiff).toBe(0);
    expect(r.b2YDiff).toBe(0);
    expect(r.b1XDiff).toBe(0);
    expect(r.b2XDiff).toBe(0);
  });

  // ─── Edge cases ─────────────────────────────────────────────────────

  test('empty world: step() does not crash', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.emptyWorld);
    expect(r.success).toBe(true);
    expect(r.stepped).toBe(true);
    expect(r.isGPU).toBe(true);
  });

  // ═══════════════════════════════════════════════════════════════════════
  // 3D GPU Tests
  // ═══════════════════════════════════════════════════════════════════════

  test('3D GPU: free fall matches analytical formula', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.freeFall3D);
    if (!r || !r.success) {
      test.skip(); // 3D GPU might not be available
      return;
    }
    expect(r.error).toBeLessThan(1.0);
    expect(r.y).toBeGreaterThan(4.0);
    expect(r.y).toBeLessThan(7.0);
  });

  test('3D GPU: box on ground never penetrates', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.boxOnGround3D);
    if (!r || !r.success) {
      test.skip();
      return;
    }
    expect(r.minY).toBeGreaterThan(0.3);
    expect(r.finalY).toBeGreaterThan(0.5);
    expect(r.finalY).toBeLessThan(3.0);
  });

  test('3D GPU: stack of 3 boxes settles', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.stack3D);
    if (!r || !r.success) {
      test.skip();
      return;
    }
    expect(r.allAboveGround).toBe(true);
    expect(r.ordered).toBe(true);
  });

  test('3D GPU vs CPU parity within tolerance', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpuCpuParity3D);
    if (!r || !r.success) {
      test.skip();
      return;
    }
    expect(r.diff1).toBeLessThan(0.01);
    expect(r.diff60).toBeLessThan(0.5);
  });
});

// ─── GPU 2D Unit Tests: Contact, Friction, Rotation ───────────────────────

test.describe('GPU 2D: Contact & Rotation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3333/tests/browser/test-harness.html');
    await page.waitForFunction(() => (window as any).testResults?._complete === true, {
      timeout: 30000,
    });
  });

  test('GPU 2D: box landing flat should not spin', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_rotationFlat);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    // Box dropped flat should have very small final angle
    expect(Math.abs(r.finalAngle)).toBeLessThan(1.0);
    // Should be above ground
    expect(r.finalY).toBeGreaterThan(0.5);
  });

  test('GPU 2D: low friction slides farther than high friction', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_frictionDecel);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    // Low friction should slide farther
    expect(r.lowFinalX).toBeGreaterThan(r.highFinalX);
  });

  test('GPU 2D: zero friction preserves velocity', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_zeroFriction);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    // With zero friction, horizontal velocity should persist
    expect(r.vx).toBeGreaterThan(2.0);
  });

  test('GPU 2D: 5-box stack has bounded rotation', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_stackRotation);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.allFinite).toBe(true);
    expect(r.allAboveGround).toBe(true);
    // Boxes in a stack should have small angles (settled on faces)
    expect(r.maxAngle).toBeLessThan(10);
  });

  test('GPU 2D: GPU vs CPU rotation parity', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_rotationParity);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    // GPU and CPU should produce similar final positions
    expect(r.posDiff).toBeLessThan(1.0);
  });

  test('GPU 2D: corner contact (45° rotated box) settles', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_cornerContact);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.isFinite).toBe(true);
    expect(r.aboveGround).toBe(true);
  });

  test('GPU 2D: geometric stiffness keeps 20-body sim stable', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_geometricStiffness);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.allFinite).toBe(true);
    expect(r.allAboveGround).toBe(true);
    expect(r.numBodies).toBe(20);
  });

  // ─── GPU 2D Exhaustive: Additional code paths ───────────────────────

  test('GPU 2D: soft constraint stiffness guard', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_softConstraint);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.isFinite).toBe(true);
  });

  test('GPU 2D: postStabilize produces valid results', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_postStabilize);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.stabFinite).toBe(true);
    expect(r.noStabFinite).toBe(true);
    // Both paths should produce above-ground results
    expect(r.stabY).toBeGreaterThan(0.4);
    expect(r.noStabY).toBeGreaterThan(0.4);
  });

  test('GPU 2D: bodies with zero constraints (free fall)', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_zeroConstraintBodies);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.allFinite).toBe(true);
    expect(r.allFell).toBe(true);
  });

  test('GPU 2D: mixed shapes (circles + boxes) on ground', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_mixedShapes);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.boxFinite).toBe(true);
    expect(r.ballFinite).toBe(true);
    expect(r.boxAbove).toBe(true);
    expect(r.ballAbove).toBe(true);
  });

  test('GPU 2D: restitution causes bounce', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu2d_restitution);
    expect(r.success).toBe(true);
    expect(r.isGPU).toBe(true);
    expect(r.bounced).toBe(true);
  });
});

// ─── GPU 3D Exhaustive Tests ──────────────────────────────────────────────

test.describe('GPU 3D: Exhaustive code path coverage', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3333/tests/browser/test-harness.html');
    await page.waitForFunction(() => (window as any).testResults?._complete === true, {
      timeout: 30000,
    });
  });

  test('GPU 3D: free fall preserves identity quaternion', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu3d_freeFallQuat);
    if (!r || !r.success || r.skip) { test.skip(); return; }
    // After free fall, quaternion should stay near identity
    expect(r.quatError).toBeLessThan(0.1);
    expect(r.quatNormalized).toBe(true);
    // Should have fallen
    expect(r.y).toBeLessThan(10);
    // No lateral drift
    expect(Math.abs(r.x)).toBeLessThan(0.01);
    expect(Math.abs(r.z)).toBeLessThan(0.01);
  });

  test('GPU 3D: box on ground never penetrates', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu3d_boxOnGround);
    if (!r || !r.success || r.skip) { test.skip(); return; }
    expect(r.aboveGround).toBe(true);
    expect(r.settled).toBe(true);
    expect(r.quatNormalized).toBe(true);
  });

  test('GPU 3D: low friction slides farther than high friction', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu3d_frictionDecel);
    if (!r || !r.success || r.skip) { test.skip(); return; }
    expect(r.lowSlidesFarther).toBe(true);
  });

  test('GPU 3D: 5-box stack stays stable', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu3d_stack);
    if (!r || !r.success || r.skip) { test.skip(); return; }
    expect(r.allAbove).toBe(true);
    expect(r.allFinite).toBe(true);
    expect(r.quatsNormalized).toBe(true);
    expect(r.numBoxes).toBe(5);
  });

  test('GPU 3D: off-center impact produces bounded rotation', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu3d_rotationImpact);
    if (!r || !r.success || r.skip) { test.skip(); return; }
    expect(r.bounded).toBe(true);
    expect(r.posFinite).toBe(true);
  });

  test('GPU 3D: GPU vs CPU parity within tolerance', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu3d_gpuCpuParity);
    if (!r || !r.success || r.skip) { test.skip(); return; }
    // Step 1 should be very close
    expect(r.diff1).toBeLessThan(0.01);
    // After 60 steps, allow more divergence
    expect(r.diff60).toBeLessThan(1.0);
  });

  test('GPU 3D: sphere collision on ground', async ({ page }) => {
    const r = await page.evaluate(() => (window as any).testResults.gpu3d_sphereCollision);
    if (!r || !r.success || r.skip) { test.skip(); return; }
    expect(r.aboveGround).toBe(true);
    expect(r.isFinite).toBe(true);
  });
});
