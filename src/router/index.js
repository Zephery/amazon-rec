import {createRouter, createWebHistory} from 'vue-router';
import ProductList from '../components/ProductList.vue';
import HelloWorld from "../components/HelloWorld.vue";
import ProductDetail from '../components/ProductDetail.vue';
import BrowsingHistory from "../components/BrowsingHistory.vue";

const routes = [
    {
        path: '/',
        name: 'HelloWorld',
        component: HelloWorld,
    },
    {
        path: '/products',
        name: 'ProductList',
        component: ProductList,
    },
    {
        path: '/product/:asin',
        name: 'ProductDetail',
        component: ProductDetail,
    },
    // 其他路由
    {
        path: "/browsing-history",
        name: "BrowsingHistory",
        component: BrowsingHistory,
    },
];

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
});

export default router;
