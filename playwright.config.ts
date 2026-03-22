import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 30000,
  retries: 1,
  use: {
    // Launch Chrome with WebGPU flags
    browserName: 'chromium',
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-angle=vulkan',
        '--disable-gpu-sandbox',
      ],
    },
  },
  webServer: {
    command: 'npx vite --port 3333 --strictPort',
    port: 3333,
    reuseExistingServer: true,
  },
});
