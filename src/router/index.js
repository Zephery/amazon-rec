import {createRouter, createWebHistory} from 'vue-router';
import ProductList from '../components/ProductList.vue';
import HelloWorld from "../components/HelloWorld.vue";
import ProductDetail from '../components/ProductDetail.vue';

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
];

const router = createRouter({
    history: createWebHistory(),
    routes,
});

export default router;
