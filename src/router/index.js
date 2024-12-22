import {createRouter, createWebHistory} from 'vue-router';
import ProductList from '../components/ProductList.vue';
import  App from '../App.vue';
import ProductDetail from '../components/ProductDetail.vue';

const routes = [
    {
        path: '/',
        name: 'Home',
        component: App,
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
];

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
});

export default router;
