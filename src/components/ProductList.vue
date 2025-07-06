<template>
  <v-container>
    <v-row>
      <v-col cols="12" class="text-center mb-4">
        <h1 class="mb-3">Amazon Products</h1>
        <v-text-field
            v-model="searchQuery"
            clearable
            label="Search products"
            flat
            solo-inverted
            @keyup.enter="handleSearch"
        >
          <template v-slot:append>
            <v-btn color="primary" @click="handleSearch" class="search-button">搜索</v-btn>

            <v-btn color="green" @click="viewBrowsingHistory" class="clear-history-button ml-2">浏览记录</v-btn>
            <v-btn color="red" @click="clearBrowsingHistory" class="clear-history-button ml-2">清理浏览记录</v-btn>
          </template>
        </v-text-field>
      </v-col>
    </v-row>

    <div
        class="product-grid"
        v-show="!loading || (loading && products.length)"
        ref="productGrid"
    >
      <v-card
          v-for="product in products"
          :key="product.asin"
          class="product-card"
          @click="viewProductDetail(product.asin)"
      >
        <div class="product-image-container">
          <img
              :src="product.imgUrl"
              :alt="product.title"
              class="product-image"
              @error="handleImageError($event, product.asin)"
          >
        </div>
        <v-card-text class="product-info pa-4">
          <div class="product-title mb-2">{{ product.title }}</div>
          <div class="price-section mb-2">
            <span class="currency">$</span>
            <span class="price">{{ product.price }}</span>
          </div>
          <div class="rating-section mb-2" style="justify-content: space-between;">
            <div style="display: flex; align-items: center;">
              <span class="stars" v-html="product.stars"></span>
              <span class="review-count">({{ product.reviews }})</span>
            </div>
            <div>
              <span class="recommend-score-badge ml-2"
                    v-if="product.re_score !== undefined && product.re_score !== null"
                    :class="{
                  'high-score': product.re_score >= 4,
                  'low-score': product.re_score <= 2
                }"
                    @mouseenter="product.showScoreTooltip = true"
                    @mouseleave="product.showScoreTooltip = false"
                    style="cursor: pointer; position: relative;"
              >
                推荐得分：{{ product.re_score.toFixed(2) }}
                <div
                    v-if="product.showScoreTooltip && (product.fine_score !== undefined || product.coarse_score !== undefined)"
                    class="score-tooltip">
                  <div v-if="product.fine_score !== undefined">粗排：{{ product.fine_score.toFixed(2) }}</div>
                  <div v-if="product.coarse_score !== undefined">精排：{{ product.coarse_score.toFixed(2) }}</div>
                </div>
              </span>
              <span
                  v-else
                  class="recommend-score-badge no-score ml-2"
                  style="cursor: default; position: relative;"
              >
                暂无推荐得分
              </span>
            </div>
          </div>
        </v-card-text>
      </v-card>
      <div ref="sentinel" class="sentinel"></div>
    </div>
    <div v-if="loading && hasMore && products.length" style="display: flex; justify-content: center;" class="my-4">
      <v-progress-circular indeterminate color="primary"/>
    </div>

    <!-- Loading more indicator -->
    <div v-if="loading && !products.length" class="text-center my-4">
      <v-progress-circular indeterminate color="primary"/>
    </div>

    <!-- No more products indicator -->
    <div v-if="!loading && !hasMore && products.length" class="text-center my-4">
      <p>No more products</p>
    </div>
  </v-container>
</template>

<script>
import axios from 'axios';
import {HTTP_REQUEST_URL} from "../../config/app.js";

export default {
  data() {
    return {
      products: [],
      currentPage: 1,
      loading: false,
      error: null,
      hasMore: true,
      observer: null,
      searchQuery: ''
    };
  },

  mounted() {
    this.fetchProducts();
    this.setupInfiniteScroll();
  },

  beforeDestroy() {
    if (this.observer) {
      this.observer.disconnect();
    }
  },

  methods: {
    // 跳转到浏览记录页面
    viewBrowsingHistory() {
      this.$router.push({name: 'BrowsingHistory'});
    },
    setupInfiniteScroll() {
      this.observer = new IntersectionObserver(
          (entries) => {
            const target = entries[0];
            if (target.isIntersecting && !this.loading && this.hasMore) {
              this.loadMore();
            }
          },
          {
            root: null,
            rootMargin: '100px',
            threshold: 0.1,
          }
      );

      if (this.$refs.sentinel) {
        this.observer.observe(this.$refs.sentinel);
      }
    },

    async fetchProducts(page = 1, query = '') {
      if (this.loading) return;

      this.loading = true;
      try {
        const response = await axios.get(HTTP_REQUEST_URL + '/products', {
          params: {
            page: page,
            per_page: 20,
            q: query
          },
        });

        if (page === 1) {
          this.products = response.data.products; // Reset products on first page
        } else {
          this.products = [...this.products, ...response.data.products]; // Append products
        }

        this.hasMore = response.data.products.length > 0;
      } catch (error) {
        this.error = 'Failed to load products';
      } finally {
        this.loading = false;
      }
    },

    loadMore() {
      if (!this.loading && this.hasMore) {
        this.currentPage += 1; // Increment page
        this.fetchProducts(this.currentPage, this.searchQuery); // Fetch more products with current search query
      }
    },

    viewProductDetail(asin) {
      this.$router.push({name: 'ProductDetail', params: {asin}});
    },

    handleImageError(event, asin) {
      event.target.src = 'https://via.placeholder.com/200'; // Fallback image
    },

    handleSearch() {
      this.currentPage = 1;
      this.hasMore = true;
      this.fetchProducts(this.currentPage, this.searchQuery);
    },
    clearBrowsingHistory() {
      const response = axios.delete(HTTP_REQUEST_URL + `/clear_clicks`);
      this.currentPage = 1;
      this.hasMore = true;
      this.fetchProducts(this.currentPage, this.searchQuery);
    }
  },
};
</script>

<style scoped>
.product-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  padding: 20px 0;
}

.product-card {
  border-radius: 8px;
  transition: all 0.3s;
  height: 100%;
  display: flex;
  flex-direction: column;
  cursor: pointer;
}

.product-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
}

.product-image-container {
  height: 200px;
  width: 100%;
  padding: 20px;
  background: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
}

.product-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  width: auto;
  height: auto;
}

.product-card:hover .product-image {
  transform: scale(1.05);
}

.product-info {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
}

.product-title {
  font-size: 14px;
  line-height: 1.4;
  height: 40px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  color: #333;
}

.price-section {
  color: #e4393c;
  font-size: 20px;
  font-weight: bold;
}

.currency {
  font-size: 14px;
  margin-right: 2px;
}

.rating-section {
  display: flex;
  align-items: center;
  gap: 8px;
}

.stars {
  color: #f7ba2a;
}

.review-count {
  color: #999;
  font-size: 12px;
}

.recommend-score-section {
  margin-top: 8px;
}

.recommend-score {
  font-size: 14px;
  font-weight: bold;
}

.high-score {
  color: #d9534f; /* Bootstrap danger color */
}

.low-score {
  color: #5cb85c; /* Bootstrap success color */
}

.no-score {
  color: #999;
}

@media (max-width: 600px) {
  .product-grid {
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 10px;
  }

  .product-title {
    font-size: 12px;
    height: 34px;
  }

  .price-section {
    font-size: 16px;
  }
}

.sentinel {
  width: 100%;
  height: 20px;
  background: transparent;
}

.score-tooltip {
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  background: rgba(0, 0, 0, 0.92);
  color: #fff;
  padding: 16px 24px;
  border-radius: 8px;
  line-height: 1.8;
  z-index: 99999;
  min-width: 160px;
  max-width: 320px;
  margin-top: 0;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
  word-break: break-all;
  text-align: left;
}
</style>
