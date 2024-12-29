import {defineConfig} from 'vite';
import vue from '@vitejs/plugin-vue';
import vuetify from 'vite-plugin-vuetify';

export default defineConfig({
    base: '/amazon-rec/', // 确保这里是你的仓库名
    plugins: [
        vue(),
        vuetify({
            autoImport: true,
        }),
    ],
});
