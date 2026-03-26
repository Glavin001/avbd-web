import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 45000,
  retries: 1,
  use: {
    // Launch Chrome with WebGPU flags
    browserName: 'chromium',
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH || undefined,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-angle=swiftshader',
        '--disable-gpu-sandbox',
        '--no-sandbox',
        '--headless=new',
      ],
    },
  },
  webServer: {
    command: 'npx vite --port 3333 --strictPort',
    port: 3333,
    reuseExistingServer: true,
  },
});
