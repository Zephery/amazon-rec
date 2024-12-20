import {createRouter, createWebHistory} from 'vue-router';
import ProductList from '../components/ProductList.vue';
import ProductDetail from '../components/ProductDetail.vue';

const routes = [
    {
        path: '/amazon-rec/products',
        name: 'ProductList',
        component: ProductList,
    },
    {
        path: '/amazon-rec/product/:asin',
        name: 'ProductDetail',
        component: ProductDetail,
    },
];

const router = createRouter({
    history: createWebHistory(),
    routes,
});

export default router;
