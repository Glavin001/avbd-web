import { test, expect } from '@playwright/test';

test('GPU radix sort with duplicate keys', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('http://localhost:3333/tests/browser/gpu-loader.html');
  await page.waitForFunction(() => (window as any).__ready === true, { timeout: 15000 });

  const result = await page.evaluate(async () => {
    const A = (window as any).AVBD;

    // Create a GPU context
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { error: 'No adapter' };
    const device = await adapter.requestDevice();
    const ctx = { device, adapter };

    // Import GpuRadixSort - it's exposed via AVBD internal
    const { GpuRadixSort } = A._internal || {};
    if (!GpuRadixSort) return { error: 'GpuRadixSort not exposed' };

    const sort = new GpuRadixSort(ctx);

    // Test with duplicate keys (simulating duplicate Morton codes)
    // 21 elements: 1 unique + 10 pairs of duplicates
    const keys = new Uint32Array([
      100,  // ground (unique)
      200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,  // 10 unique positions
      200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,  // 10 duplicates
    ]);
    const vals = new Uint32Array([
      0,  // ground
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // first 10 bodies
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  // second 10 bodies
    ]);

    const result = await sort.sortAndReadback(keys, vals);
    sort.destroy();
    device.destroy();

    return {
      sortedKeys: Array.from(result.sortedKeys),
      sortedVals: Array.from(result.sortedVals),
      n: keys.length,
    };
  });

  console.log('Sorted keys:', JSON.stringify(result));
});
