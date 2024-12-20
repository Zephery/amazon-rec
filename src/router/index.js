import {createRouter, createWebHistory} from 'vue-router';
import ProductList from '../components/ProductList.vue';
import ProductDetail from '../components/ProductDetail.vue';

const routes = [
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
    history: createWebHistory(),
    routes,
});

export default router;
