import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: {
    port: 3000,
    open: '/demo-2d.html',
  },
  resolve: {
    alias: {
      '@': '../src',
    },
  },
});
