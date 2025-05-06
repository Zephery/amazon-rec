<template>
  <v-container>
    <!-- 顶部标题和返回按钮 -->
    <v-row>
      <v-col cols="12" class="d-flex align-center justify-start mb-4">
        <v-btn
            class="back-button mr-3"
            color="primary"
            dark
            elevation="2"
            @click="$router.go(-1)"
        >← Back</v-btn>
        <h1 class="mb-0">Browsing History</h1>
      </v-col>
    </v-row>

    <!-- 加载状态提示 -->
    <div v-if="loading" class="loading-container">
      <div class="spinner"></div>
      <p>Loading data, please wait...</p>
    </div>

    <!-- 浏览记录展示 -->
    <div v-else>
      <div class="product-list">
        <v-card
            v-for="product in browsingHistory"
            :key="product.asin"
            class="product-list-item"
            @click="viewProductDetail(product.asin)"
        >
          <div class="product-list-container">
            <img
                :src="product.imgUrl"
                :alt="product.title"
                class="product-image"
                @error="handleImageError($event, product.asin)"
            />
            <v-card-text class="product-info">
              <div class="product-title">{{ product.title }}</div>
              <div class="click-time">
                <span>Clicked on:</span>
                <span>{{ formatTimestamp(product.click_time) }}</span>
              </div>
              <div class="price-section">
                <span class="currency">$</span>
                <span class="price">{{ product.price }}</span>
              </div>
              <div class="rating-section">
                <span class="stars" v-html="product.stars"></span>
                <span class="review-count">({{ product.reviews }})</span>
              </div>
            </v-card-text>
          </div>
        </v-card>
      </div>

      <!-- 无数据提示 -->
      <div v-if="!browsingHistory.length" class="text-center my-4">
        <p>You haven't browsed any products yet!</p>
      </div>
    </div>
  </v-container>
</template>

<script>
import axios from "axios";
import { HTTP_REQUEST_URL } from "../../config/app.js";

export default {
  data() {
    return {
      browsingHistory: [],
      loading: false, // 用于控制加载动画的显示
    };
  },

  mounted() {
    this.loadBrowsingHistory();
  },

  methods: {
    async loadBrowsingHistory() {
      try {
        // 开启加载状态
        this.loading = true;

        // 模拟接口调用
        const response = await axios.get(HTTP_REQUEST_URL + "/get_clicks");
        this.browsingHistory = response.data.products || [];
      } catch (error) {
        console.error("Failed to fetch browsing history:", error);
      } finally {
        // 请求结束后关闭加载状态
        this.loading = false;
      }
    },

    viewProductDetail(asin) {
      this.$router.push({ name: "ProductDetail", params: { asin } });
    },

    handleImageError(event, asin) {
      event.target.src = "https://via.placeholder.com/200"; // Fallback image
    },

    formatTimestamp(timestamp) {
      const date = new Date(timestamp * 1000);
      const yyyy = date.getFullYear();
      const MM = String(date.getMonth() + 1).padStart(2, "0");
      const dd = String(date.getDate()).padStart(2, "0");
      const HH = String(date.getHours()).padStart(2, "0");
      const mm = String(date.getMinutes()).padStart(2, "0");
      const ss = String(date.getSeconds()).padStart(2, "0");
      return `${yyyy}-${MM}-${dd} ${HH}:${mm}:${ss}`;
    },
  },
};
</script>
<style scoped>
/* 返回按钮样式 */
.back-button {
  font-size: 16px;
  padding: 8px 16px;
  border-radius: 25px;
  background-color: #007bff;
  color: #ffffff;
  text-transform: none;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
}

.back-button:hover {
  background-color: #0056b3;
  box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
  transform: translateY(-2px);
}

.back-button:active {
  background-color: #004080;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  transform: translateY(0);
}

/* 加载状态样式 */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.spinner {
  border: 6px solid #f3f3f3; /* 灰色外圈 */
  border-top: 6px solid #007bff; /* 蓝色转动部分 */
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.loading-container p {
  font-size: 14px;
  margin-top: 10px;
  color: #666;
}

/* 产品列表样式 */
.product-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.product-list-item {
  display: flex;
  flex-direction: row;
  padding: 16px;
  align-items: center;
  cursor: pointer;
  transition: all 0.3s;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.product-list-item:hover {
  transform: translateY(-2px);
}

.product-list-container {
  display: flex;
  gap: 15px;
  align-items: center;
}

.product-image {
  width: 120px;
  height: 120px;
  object-fit: contain;
  border-radius: 8px;
  background: #f4f4f4;
}

.product-info {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.click-time {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.click-time span {
  font-weight: bold;
}

.product-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 8px;
  color: #333;
  line-height: 1.4;
}

.price-section {
  font-size: 18px;
  font-weight: bold;
  color: #e4393c;
  margin-bottom: 8px;
}

.currency {
  font-size: 14px;
  margin-right: 4px;
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

