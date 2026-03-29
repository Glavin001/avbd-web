import { test, expect } from '@playwright/test';

// Test the key failing cases
for (const n of [5, 10, 15, 20, 50]) {
  test(`GPU collision ${n} overlapping bodies`, async ({ page }) => {
    test.setTimeout(60000);
    await page.goto('http://localhost:3333/tests/browser/gpu-loader.html');
    await page.waitForFunction(() => (window as any).__ready === true, { timeout: 15000 });

    const result = await page.evaluate(async (bodyCount: number) => {
      const A = (window as any).AVBD;
      const w1 = new A.World({ x: 0, y: -9.81 }, { iterations: 4, useGPUCollision: true });
      w1.createCollider(A.ColliderDesc.cuboid(30, 0.5).setFriction(0.5));
      const w2 = new A.World({ x: 0, y: -9.81 }, { iterations: 4, useGPUCollision: false });
      w2.createCollider(A.ColliderDesc.cuboid(30, 0.5).setFriction(0.5));
      for (let i = 0; i < bodyCount; i++) {
        const x = (i % 10) * 1.5 - 7.5;
        const y = 1.0;
        w1.createCollider(A.ColliderDesc.cuboid(0.5, 0.5).setFriction(0.5),
          w1.createRigidBody(A.RigidBodyDesc.dynamic().setTranslation(x, y)));
        w2.createCollider(A.ColliderDesc.cuboid(0.5, 0.5).setFriction(0.5),
          w2.createRigidBody(A.RigidBodyDesc.dynamic().setTranslation(x, y)));
      }
      try { await w1.step(); } catch (e) { return { error: (e as Error).message }; }
      w2.stepCPU();
      return {
        gpuPairs: w1.lastTimings?.numPairs ?? -1,
        gpuCr: w1.lastTimings?.numConstraints ?? -1,
        cpuCr: w2.lastTimings?.numConstraints ?? -1,
      };
    }, n);
    console.log(`n=${n}: ${JSON.stringify(result)}`);
    if ('error' in result) {
      expect(result.error).toBeUndefined();
    } else {
      expect(result.gpuCr).toBe(result.cpuCr);
    }
  });
}

// Test grid layouts
for (const [cols, rows] of [[5, 2], [10, 2], [10, 3]]) {
  test(`GPU collision grid ${cols}x${rows}`, async ({ page }) => {
    test.setTimeout(60000);
    await page.goto('http://localhost:3333/tests/browser/gpu-loader.html');
    await page.waitForFunction(() => (window as any).__ready === true, { timeout: 15000 });

    const result = await page.evaluate(async ([cols, rows]: number[]) => {
      const A = (window as any).AVBD;
      const n = cols * rows;
      const w1 = new A.World({ x: 0, y: -9.81 }, { iterations: 4, useGPUCollision: true });
      w1.createCollider(A.ColliderDesc.cuboid(30, 0.5).setFriction(0.5));
      const w2 = new A.World({ x: 0, y: -9.81 }, { iterations: 4, useGPUCollision: false });
      w2.createCollider(A.ColliderDesc.cuboid(30, 0.5).setFriction(0.5));
      for (let i = 0; i < n; i++) {
        const x = (i % cols) * 1.5;
        const row = Math.floor(i / cols);
        const y = 1.0 + row * 1.0;
        w1.createCollider(A.ColliderDesc.cuboid(0.5, 0.5).setFriction(0.5),
          w1.createRigidBody(A.RigidBodyDesc.dynamic().setTranslation(x, y)));
        w2.createCollider(A.ColliderDesc.cuboid(0.5, 0.5).setFriction(0.5),
          w2.createRigidBody(A.RigidBodyDesc.dynamic().setTranslation(x, y)));
      }
      try { await w1.step(); } catch (e) { return { error: (e as Error).message }; }
      w2.stepCPU();
      return {
        gpuPairs: w1.lastTimings?.numPairs ?? -1,
        gpuCr: w1.lastTimings?.numConstraints ?? -1,
        cpuCr: w2.lastTimings?.numConstraints ?? -1,
      };
    }, [cols, rows]);
    console.log(`${cols}x${rows}: ${JSON.stringify(result)}`);
    if ('error' in result) {
      expect(result.error).toBeUndefined();
    } else {
      expect(result.gpuCr).toBe(result.cpuCr);
    }
  });
}
