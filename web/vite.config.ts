import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: process.env.FORGE_BASE || '/Forge/',
  plugins: [react()],
  worker: { format: 'es' },
  build: { target: 'es2022', rollupOptions: { input: { main: 'index.html', validation: 'validation.html' } } },
});
