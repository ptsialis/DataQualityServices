import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Automatisch erkennen, ob Docker oder lokal
const isDocker = process.env.DOCKER === 'true';
const backendTarget = isDocker ? 'http://backend:5000' : 'http://localhost:5000';

// ---- Vite Config ----
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/upload': backendTarget,
      '/status': backendTarget,
      '/original': backendTarget,
      '/datagraphs': backendTarget,
      '/inference': backendTarget,
      '/imputation': backendTarget,
      '/anomaly': backendTarget,
      '/personal': backendTarget,
      '/metadata': backendTarget,
      '/summary': backendTarget,
      '/detectedCounts': backendTarget,
      '/model': backendTarget,
      '/cleaned': backendTarget,
      '/downloadZip': backendTarget,
      '/reset': backendTarget
    }
  }
});
