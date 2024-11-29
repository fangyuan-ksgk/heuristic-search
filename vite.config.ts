import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Add any aliases if needed
    },
  },
  server: {
    port: 3000,
  },
  optimizeDeps: {
    exclude: ['node:fs/promises'] // Exclude Node.js built-in modules
  }
}) 