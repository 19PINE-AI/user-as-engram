import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Deployed under https://ring0.me/research/user-as-engram/ — base must match
// so asset and data (import.meta.env.BASE_URL) paths resolve under the subpath.
export default defineConfig({
  base: '/research/user-as-engram/',
  plugins: [react()],
})
