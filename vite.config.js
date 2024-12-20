import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import vuetify from 'vite-plugin-vuetify';

export default defineConfig({
  base: process.env.NODE_ENV === 'production' ? '/amazon-rec' : '/',
  plugins: [
    vue(),
    vuetify({
      autoImport: true,
    }),
  ],
});
