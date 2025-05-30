import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import vuetify from 'vite-plugin-vuetify';

export default defineConfig({
  server: {
    watch: {
      ignored: ['**/node_modules/**', '**/dist/**','**/rec-flask/**']
    }
  },
  plugins: [
    vue(),
    vuetify({
      autoImport: true,
    }),
  ],
});
