<template>
  <v-container>
    <v-row>
      <v-col cols="12" class="text-center mb-4">
        <h1 class="mb-3">Browsing History</h1>
      </v-col>
    </v-row>

    <div class="product-grid">
      <v-card
          v-for="product in browsingHistory"
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
          />
        </div>
        <v-card-text class="product-info pa-4">
          <div class="product-title mb-2">{{ product.title }}</div>
          <div class="price-section mb-2">
            <span class="currency">$</span>
            <span class="price">{{ product.price }}</span>
          </div>
          <div class="rating-section mb-2">
            <span class="stars" v-html="product.stars"></span>
            <span class="review-count">({{ product.reviews }})</span>
          </div>
        </v-card-text>
      </v-card>
    </div>

    <div v-if="!browsingHistory.length" class="text-center my-4">
      <p>You haven't browsed any products yet!</p>
    </div>
  </v-container>
</template>

<script>
import axios from "axios";
import {HTTP_REQUEST_URL} from "../../config/app.js";

export default {
  data() {
    return {
      browsingHistory: [],
    };
  },

  mounted() {
    this.loadBrowsingHistory();
  },

  methods: {
    async loadBrowsingHistory() {
      try {
        // 从后端获取浏览记录
        const response = await axios.get(HTTP_REQUEST_URL+"/get_clicks");
        this.browsingHistory = response.data.products || [];
      } catch (error) {
        console.error("Failed to fetch browsing history:", error);
      }
    },

    viewProductDetail(asin) {
      this.$router.push({ name: "ProductDetail", params: { asin } });
    },

    handleImageError(event, asin) {
      event.target.src = "https://via.placeholder.com/200"; // Fallback image
    },
  },
};
</script>

<style scoped>
/* 重用与主页面相同的样式 */
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
</style>
