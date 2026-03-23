import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  build: {
    outDir: '../demo-dist',
    emptyOutDir: true,
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        demo2d: resolve(__dirname, 'demo-2d.html'),
        demo3d: resolve(__dirname, 'demo-3d.html'),
      },
    },
  },
  server: {
    port: 3000,
    open: '/',
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, '../src'),
    },
  },
});
