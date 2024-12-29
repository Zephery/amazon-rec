<script setup lang="ts">
import {RouterLink, RouterView} from 'vue-router'
import HelloWorld from './components/HelloWorld.vue'
import ProductList from './components/ProductList.vue'
</script>

<template>
  <div>
    <el-button type="primary">我是按钮哦</el-button>
  </div>

  <ProductList/>
  <main class="flex justify-center items-center flex-col px-[20px] gap-4">
    <div class="text-xl text-sky-500">欢迎使用 random-number-system</div>
    <p class="text-gray-500 text-sm">
      它可以自动生成给定范围内的随机正整数，可用帮你打乱顺序记东西：比如记英语单词~
    </p>
  </main>
  <header>
    <img alt="Vue logo" class="logo" src="@/assets/logo.svg" width="125" height="125"/>

    <div class="wrapper bg-yellow-500">
      <HelloWorld msg="You did it!"/>

      <nav>
        <RouterLink to="/">Home</RouterLink>
        <RouterLink to="/about">About</RouterLink>
      </nav>
    </div>
  </header>

  <RouterView/>
</template>

<style scoped>
header {
  line-height: 1.5;
  max-height: 100vh;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}

nav {
  width: 100%;
  font-size: 12px;
  text-align: center;
  margin-top: 2rem;
}

nav a.router-link-exact-active {
  color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }

  nav {
    text-align: left;
    margin-left: -1rem;
    font-size: 1rem;

    padding: 1rem 0;
    margin-top: 1rem;
  }
}
</style>