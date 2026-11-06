import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()], build:{rollupOptions:{output:{manualChunks:{charts:['recharts']}}}}, server: {host:'127.0.0.1', port:5173, strictPort:true, proxy:{'/api':'http://127.0.0.1:8000'}}, test:{include:['tests/**/*.test.tsx'],environment:'jsdom',setupFiles:['./tests/setup.ts'],css:false} });
