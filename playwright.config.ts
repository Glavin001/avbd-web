import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 60_000,
  retries: 1,
  use: {
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
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npx vite --port 3333 --strictPort',
    port: 3333,
    reuseExistingServer: true,
  },
});
