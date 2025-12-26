import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  plugins: [wasm()],
  build: {
    target: 'esnext' // WebAssembly requires modern browser support
  },
  server: {
    fs: {
      allow: ['..']
    }
  }
});
