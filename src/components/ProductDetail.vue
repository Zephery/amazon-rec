<template>
  <v-container>
    <v-row>
      <v-col cols="12" md="6">
        <!-- 商品图片区域 -->
        <div class="image-container">
          <img :src="product.imgUrl" :alt="product.title" class="product-image">
        </div>
      </v-col>

      <v-col cols="12" md="6">
        <!-- 商品信息区域 -->
        <div class="product-info">
          <h1 class="product-title">{{ product.title }}</h1>

          <!-- 价格信息 -->
          <div class="price-section">
            <div class="current-price">
              <span class="currency">$</span>
              <span class="amount">{{ product.price }}</span>
            </div>
            <div class="original-price" v-if="product.listPrice">
              原价: <span class="strike-through">${{ product.listPrice }}</span>
            </div>
            <div class="savings" v-if="product.listPrice">
              节省: ${{ (product.listPrice - product.price).toFixed(2) }}
            </div>
          </div>

          <!-- 评分和销量 -->
          <div class="rating-section">
            <div class="stars">
              <v-rating
                  :model-value="Number(product.stars)"
                  color="amber"
                  density="compact"
                  half-increments
                  readonly
              ></v-rating>
              <span class="rating-text">{{ product.stars }} 星</span>
            </div>
            <div class="reviews" v-if="product.reviews">
              {{ product.reviews }} 条评价
            </div>
            <div class="monthly-sales" v-if="product.boughtInLastMonth">
              近30天销量: {{ product.boughtInLastMonth }}
            </div>
          </div>

          <!-- 附加信息 -->
          <div class="additional-info">
            <v-list>
              <v-list-item v-if="additionalInfo.shipping">
                <template v-slot:prepend>
                  <v-icon color="primary">mdi-truck-delivery</v-icon>
                </template>
                <v-list-item-title>{{ additionalInfo.shipping }}</v-list-item-title>
              </v-list-item>

              <v-list-item v-if="additionalInfo.return_policy">
                <template v-slot:prepend>
                  <v-icon color="primary">mdi-undo</v-icon>
                </template>
                <v-list-item-title>{{ additionalInfo.return_policy }}</v-list-item-title>
              </v-list-item>

              <v-list-item v-if="additionalInfo.warranty">
                <template v-slot:prepend>
                  <v-icon color="primary">mdi-shield-check</v-icon>
                </template>
                <v-list-item-title>{{ additionalInfo.warranty }}</v-list-item-title>
              </v-list-item>
            </v-list>
          </div>

          <!-- 购买按钮 -->
          <div class="action-buttons">
            <v-btn
                color="primary"
                block
                size="large"
                :href="product.productURL"
                target="_blank"
                class="mb-4"
            >
              <v-icon left>mdi-cart</v-icon>
              在亚马逊购买
            </v-btn>

            <v-btn
                variant="outlined"
                block
                @click="goBack"
            >
              返回列表
            </v-btn>
          </div>
        </div>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import axios from 'axios';
import {HTTP_REQUEST_URL} from '../../config/app';

export default {
  data() {
    return {
      product: {},
      additionalInfo: {},
    };
  },
  async created() {
    const {asin} = this.$route.params;
    try {
      const response = await axios.get(HTTP_REQUEST_URL + `/products/${asin}`);
      this.product = response.data.data.product;
      this.additionalInfo = response.data.data.additional_info;
    } catch (error) {
      console.error('Error fetching product details:', error);
    }
  },
  methods: {
    goBack() {
      this.$router.back();
    },
  },
};
</script>

<style scoped>
.image-container {
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  height: 400px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.product-image {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}

.product-info {
  padding: 20px;
}

.product-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #2c3e50;
  line-height: 1.3;
}

.price-section {
  margin-bottom: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.current-price {
  font-size: 28px;
  color: #e41e31;
  font-weight: 700;
  margin-bottom: 4px;
}

.currency {
  font-size: 20px;
  margin-right: 4px;
}

.original-price {
  color: #666;
  font-size: 14px;
  margin-bottom: 4px;
}

.strike-through {
  text-decoration: line-through;
}

.savings {
  color: #2ecc71;
  font-weight: 600;
  font-size: 14px;
  margin-top: 2px;
}

.rating-section {
  margin-bottom: 20px;
  padding: 2px 0;
  background: #f8f9fa;
  border-radius: 8px;
}

.stars {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.rating-text {
  margin-left: 8px;
  color: #666;
}

.reviews, .monthly-sales {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
}

.additional-info {
  margin-bottom: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.action-buttons {
  margin-top: 16px;
}

@media (max-width: 600px) {
  .product-title {
    font-size: 20px;
  }

  .current-price {
    font-size: 24px;
  }
}
</style>
