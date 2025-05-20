from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# 假设有一个商品广告数据库，用字典模拟：商品 ID、预算、出价等
ads_db = [
    {"id": 1, "name": "手机1", "budget": 100, "bid": 1.5, "target_region": "Beijing", "ctr": 0.03},
    {"id": 2, "name": "手机2", "budget": 50, "bid": 1.2, "target_region": "Shanghai", "ctr": 0.02},
    {"id": 3, "name": "耳机3", "budget": 20, "bid": 0.8, "target_region": "Beijing", "ctr": 0.05},
    {"id": 4, "name": "音响4", "budget": 10, "bid": 0.5, "target_region": "Shenzhen", "ctr": 0.01},
]

# 用户行为数据和上下文（用于定向和个性化）
user_data = {
    "region": "Beijing",  # 用户所在区域
    "recent_ads_clicked": [1, 3],  # 用户点击过的广告 ID（用于频次和个性化调控）
}

# 曝光计数，用于频次限制
exposure_count = {}

# 配置：频次限制（每个广告每天最多对同一用户展示3次）
FREQUENCY_CAP = 3


@app.route('/get_ads', methods=['GET'])
def get_ads():
    """
    API: 返回可以曝光的广告候选集
    """
    # Step 1: 获取用户上下文
    user_region = user_data.get("region")

    # Step 2: 召回：筛选出满足定向规则的广告
    recalled_ads = []
    for ad in ads_db:
        # 地区定向：仅召回用户所在区域的广告
        if ad["target_region"] == user_region and ad["budget"] > 0:
            recalled_ads.append(ad)

    if not recalled_ads:
        return jsonify({"message": "No ads available for display."})

    # Step 3: 预估与排序
    # 计算 eCPM = CTR * 出价，并按 eCPM 高低排序
    for ad in recalled_ads:
        ad["eCPM"] = ad["ctr"] * ad["bid"]
    sorted_ads = sorted(recalled_ads, key=lambda x: x["eCPM"], reverse=True)

    # Step 4: 过滤频次：检查广告是否超出频次限制
    final_ads = []
    for ad in sorted_ads:
        ad_id = ad["id"]
        exposure_count.setdefault(ad_id, 0)  # 初始化曝光计数
        if exposure_count[ad_id] < FREQUENCY_CAP:
            final_ads.append(ad)
            exposure_count[ad_id] += 1  # 增加曝光计数

    # Step 5: 更新预算、返回最终曝光广告
    for ad in final_ads:
        ad["budget"] -= ad["bid"]  # 扣除广告预算

    return jsonify({"exposed_ads": final_ads})


@app.route('/reset', methods=['POST'])
def reset_counts():
    """
    重置数据：每天清空曝光计数（频次上限逻辑）
    """
    global exposure_count
    exposure_count = {ad["id"]: 0 for ad in ads_db}
    return jsonify({"message": "Exposure counts reset."})


if __name__ == '__main__':
    app.run(debug=True)