"""Test for useraes backend app - EricasZ"""
import os
import io
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, render_template, jsonify, request
from PIL import Image
import requests

app = Flask(__name__)

# ================= 加载外部配置 =================
_config = {}
_config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
if os.path.exists(_config_path):
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config = json.load(f)

# ================= 配置与全局变量 =================

USE_LOCAL_IMAGES = True
TEST_IMAGES = [
    {"id": "img_01", "url": "https://picsum.photos/seed/1/400/300"},
    {"id": "img_02", "url": "https://picsum.photos/seed/2/400/300"},
    {"id": "img_03", "url": "https://picsum.photos/seed/3/400/300"},
    {"id": "img_04", "url": "https://picsum.photos/seed/4/400/300"},
    {"id": "img_05", "url": "https://picsum.photos/seed/5/400/300"},
    {"id": "img_06", "url": "https://picsum.photos/seed/6/400/300"},
    {"id": "img_07", "url": "https://picsum.photos/seed/7/400/300"},
    {"id": "img_08", "url": "https://picsum.photos/seed/8/400/300"},
    {"id": "img_09", "url": "https://picsum.photos/seed/9/400/300"},
    {"id": "img_10", "url": "https://picsum.photos/seed/10/400/300"},
    {"id": "img_11", "url": "https://picsum.photos/seed/11/400/300"},
    {"id": "img_12", "url": "https://picsum.photos/seed/12/400/300"},
]

# 本地图片目录（从配置文件读取）
_local_images_dir = _config.get('useraes', {}).get('local_images_dir', '')
if _local_images_dir and os.path.exists(_local_images_dir):
    LOCAL_IMAGES = [os.path.join(_local_images_dir, f) for f in os.listdir(_local_images_dir)]
else:
    LOCAL_IMAGES = []
    USE_LOCAL_IMAGES = False

PROFILE_SAVE_PATH = _config.get('useraes', {}).get('profile_path', "user_profile.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型加载与特征提取 =================
# 加载预训练 ResNet18，并移除最后的全连接层，只保留特征提取部分
print(f"[初始化] 正在加载预训练模型 (Device: {DEVICE})...")
print(f"[初始化] 使用设备: {DEVICE}")
try:
    print("[初始化] 开始加载 ResNet18 预训练权重...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("[初始化] ResNet18 权重加载完成")
    
    # 移除最后的 AvgPool 和 FC 层，使用 AdaptiveAvgPool2d 确保输出维度固定
    print("[初始化] 构建特征提取网络...")
    modules = list(base_model.children())[:-2]
    feature_extractor = nn.Sequential(*modules)
    print(f"[初始化] 特征提取网络构建完成，层数: {len(modules)}")
    
    feature_extractor.to(DEVICE)
    print(f"[初始化] 模型已移动到 {DEVICE}")
    
    feature_extractor.eval()
    print("[初始化] 模型设置为评估模式")
    print("[初始化] 模型加载成功。")
except Exception as e:
    print(f"[错误] 模型加载失败：{e}")
    feature_extractor = None

# 图像预处理 (必须与评分代码保持一致)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_feature_vector(image_input):
    """
    输入：PIL Image 或 图像路径
    输出：归一化的 numpy 特征向量 (512 维)
    """
    if feature_extractor is None:
        raise RuntimeError("模型未加载")

    print(f"[特征提取] 开始处理图像输入类型: {type(image_input).__name__}")
    
    if isinstance(image_input, str):
        print(f"[特征提取] 从路径加载图片: {image_input}")
        image = Image.open(image_input).convert('RGB')
    else:
        print("[特征提取] 处理 PIL Image 对象")
        image = image_input.convert('RGB')
    
    print(f"[特征提取] 图片原始尺寸: {image.size}")
    print("[特征提取] 应用预处理变换...")
    
    input_tensor = preprocess(image)
    print(f"[特征提取] 预处理后张量形状: {input_tensor.shape}")
    
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    print(f"[特征提取] 批次张量形状: {input_batch.shape}, 设备: {input_batch.device}")

    with torch.no_grad():
        print("[特征提取] 开始特征提取...")
        # 提取特征
        features = feature_extractor(input_batch)
        print(f"[特征提取] 卷积特征形状: {features.shape}")
        
        # 全局平均池化将 (1, 512, 7, 7) 变为 (1, 512, 1, 1)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        print(f"[特征提取] 池化后特征形状: {features.shape}")
        
        # 展平
        features = features.squeeze()
        print(f"[特征提取] 展平后特征形状: {features.shape}")

    # 转换为 numpy 并 L2 归一化 (便于计算余弦相似度)
    print("[特征提取] 转换为 numpy 数组并归一化...")
    feature_np = features.cpu().numpy()
    print(f"[特征提取] numpy 数组形状: {feature_np.shape}")
    
    norm = np.linalg.norm(feature_np)
    print(f"[特征提取] 特征向量 L2 范数: {norm:.6f}")
    
    if norm > 0:
        feature_np /= norm
        print("[特征提取] L2 归一化完成")
    else:
        print("[警告] 特征向量范数为0，无法归一化")

    print("[特征提取] 特征提取完成")
    return feature_np


def download_image_from_url(url):
    try:
        print(f"[下载] 正在下载图片: {url}")
        response = requests.get(url, timeout=10)
        print(f"[下载] 图片下载完成，状态码: {response.status_code}")
        image = Image.open(io.BytesIO(response.content))
        print(f"[下载] 图片解析完成，尺寸: {image.size}")
        return image
    except Exception as e:
        print(f"[错误] 下载图片失败 {url}: {e}")
        return None


# ================= 路由接口 =================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/get_test_images', methods=['GET'])
def get_test_images():
    """返回测试图片列表"""
    if USE_LOCAL_IMAGES:
        print(f"[API] 使用本地图片模式，返回 {len(LOCAL_IMAGES)} 张图片")
        # 为本地图片生成统一的 ID 格式和 URL
        images = []
        for i, path in enumerate(LOCAL_IMAGES):
            img_id = f"local_{i+1:02d}"
            # 使用文件名作为 URL 参数，后端会通过 /api/local_image/ 路由提供
            img_url = f"/api/local_image/{img_id}"
            images.append({"id": img_id, "url": img_url, "path": path})
        return jsonify({"images": images})
    else:
        print(f"[API] 使用网络图片模式，返回 {len(TEST_IMAGES)} 张图片")
        return jsonify({"images": TEST_IMAGES})


@app.route('/api/local_image/<image_id>')
def serve_local_image(image_id):
    """提供本地图片文件给前端访问"""
    if not USE_LOCAL_IMAGES:
        return jsonify({"error": "当前不是本地图片模式"}), 400
    
    # 从 image_id 提取索引 (如 local_01 -> 0)
    try:
        img_index = int(image_id.split('_')[1]) - 1
    except (ValueError, IndexError):
        return jsonify({"error": "无效的图片 ID"}), 400
    
    if 0 <= img_index < len(LOCAL_IMAGES):
        img_path = LOCAL_IMAGES[img_index]
        print(f"[图片服务] 请求本地图片：{img_path}")
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"[图片服务] 错误：文件不存在 {img_path}")
            return jsonify({"error": f"文件不存在：{img_path}"}), 404
        
        try:
            # 读取并返回图片
            with open(img_path, 'rb') as f:
                image_data = f.read()
            print(f"[图片服务] 成功返回图片：{img_path}")
            return app.response_class(image_data, mimetype='image/jpeg')
        except Exception as e:
            print(f"[图片服务] 错误：读取文件失败 {e}")
            return jsonify({"error": str(e)}), 500
    else:
        print(f"[图片服务] 错误：未找到图片索引 {img_index}")
        return jsonify({"error": "图片不存在"}), 404


@app.route('/api/submit_profile', methods=['POST'])
def submit_profile():
    """
    接收用户评分，计算审美向量，保存为 pickle
    """
    print("[API] 收到用户审美数据提交请求")
    data = request.json
    ratings = data.get('ratings', [])
    
    print(f"[API] 接收到 {len(ratings)} 条评分数据")

    if not ratings:
        print("[API] 错误: 未收到评分数据")
        return jsonify({"success": False, "message": "未收到评分数据"})

    weighted_vectors = []
    weights_sum = 0
    successful_count = 0
    failed_count = 0

    print("[处理] 开始处理用户审美数据...")
    
    print(f"[处理] 总共需要处理 {len(ratings)} 张图片")

    for idx, item in enumerate(ratings, 1):
        img_id = item['id']
        score = item['score']
        
        print(f"[处理] [{idx}/{len(ratings)}] 处理图片 {img_id}, 评分：{score}")

        # 根据模式获取图片路径或 URL
        if USE_LOCAL_IMAGES:
            # 本地模式：从 local_images 列表中查找
            img_index = int(img_id.split('_')[1]) - 1  # 从 local_01, local_02...提取索引
            if 0 <= img_index < len(LOCAL_IMAGES):
                img_path = LOCAL_IMAGES[img_index]
                print(f"[处理] [{idx}/{len(ratings)}] 使用本地图片路径：{img_path}")
                # 从本地路径加载图片
                try:
                    pil_img = Image.open(img_path).convert('RGB')
                    print(f"[处理] [{idx}/{len(ratings)}] 本地图片加载成功")
                except Exception as e:
                    print(f"[错误] [{idx}/{len(ratings)}] 本地图片加载失败 {img_path}: {e}")
                    failed_count += 1
                    continue
            else:
                print(f"[错误] [{idx}/{len(ratings)}] 未找到本地图片 {img_id}")
                failed_count += 1
                continue
        else:
            # 网络模式：从 TEST_IMAGES 中查找 URL
            img_url = next((img['url'] for img in TEST_IMAGES if img['id'] == img_id), None)
            
            if img_url:
                print(f"[处理] [{idx}/{len(ratings)}] 找到图片 URL: {img_url}")
                # 从网络下载图片
                pil_img = download_image_from_url(img_url)
                if pil_img:
                    print(f"[处理] [{idx}/{len(ratings)}] 图片下载成功")
                else:
                    print(f"[错误] [{idx}/{len(ratings)}] 图片下载失败 {img_id}")
                    failed_count += 1
                    continue
            else:
                print(f"[错误] [{idx}/{len(ratings)}] 未找到图片 {img_id} 的 URL")
                failed_count += 1
                continue
        
        # 如果图片加载成功，继续特征提取
        if pil_img:
            # 2. 提取特征
            try:
                vec = extract_feature_vector(pil_img)
                print(f"[处理] [{idx}/{len(ratings)}] 特征提取成功，特征维度：{vec.shape}")
                # 3. 加权 (分数越高，权重越大)
                weighted_vec = vec * score
                weighted_vectors.append(weighted_vec)
                weights_sum += score
                successful_count += 1
                print(f"[处理] [{idx}/{len(ratings)}] 加权处理完成，当前权重和：{weights_sum:.2f}")
            except Exception as e:
                print(f"[错误] [{idx}/{len(ratings)}] 特征提取失败 {img_id}: {e}")
                failed_count += 1

    print(f"[处理] 处理完成 - 成功: {successful_count}, 失败: {failed_count}")
    
    if len(weighted_vectors) == 0:
        print("[API] 错误: 没有成功的图片可处理")
        return jsonify({"success": False, "message": "没有成功的图片可处理"})

    print("[计算] 开始计算加权平均向量 (用户审美向量)...")
    print(f"[计算] 加权向量数量: {len(weighted_vectors)}, 权重总和: {weights_sum:.2f}")
    
    # 4. 计算加权平均向量 (用户审美向量)
    user_preference_vector = np.sum(weighted_vectors, axis=0) / weights_sum
    print(f"[计算] 用户审美向量计算完成，维度: {user_preference_vector.shape}")

    # 再次归一化用户向量
    print("[计算] 对用户审美向量进行L2归一化...")
    norm = np.linalg.norm(user_preference_vector)
    print(f"[计算] 归一化前向量范数: {norm:.6f}")
    
    if norm > 0:
        user_preference_vector /= norm
        print("[计算] L2归一化完成")
    else:
        print("[警告] 向量范数为0，无法归一化")

    # 5. 保存数据
    print(f"[保存] 准备保存用户审美模型到: {PROFILE_SAVE_PATH}")
    profile_data = {
        "vector": user_preference_vector,
        "version": "1.0",
        "method": "weighted_avg_resnet18",
        "statistics": {
            "total_ratings": len(ratings),
            "successful_processing": successful_count,
            "failed_processing": failed_count,
            "weight_sum": weights_sum
        }
    }

    print("[保存] 开始序列化数据...")
    with open(PROFILE_SAVE_PATH, 'wb') as f:
        pickle.dump(profile_data, f)  # type: ignore
    print("[保存] 数据序列化完成")

    save_path = os.path.abspath(PROFILE_SAVE_PATH)
    print(f"[保存] 审美模型已保存至：{save_path}")

    return jsonify({
        "success": True,
        "message": "保存成功",
        "file_path": save_path,
        "statistics": {
            "total_processed": len(ratings),
            "successful": successful_count,
            "failed": failed_count
        }
    })


if __name__ == '__main__':
    print("[启动] ==================== AutoCull 用户审美测试系统 ====================")
    print("[启动] 系统初始化开始...")
    
    # 确保 templates 文件夹存在，如果 index.html 在同级目录请调整
    if not os.path.exists('templates'):
        print("[启动] 创建 templates 目录...")
        os.makedirs('templates')
        # 为了演示方便，如果用户没创建 templates 文件夹，这里可以提示
        print("[警告] 未找到 templates 文件夹，已自动创建，请确保 index.html 位于 templates 文件夹内。")
    else:
        print("[启动] templates 目录已存在")

    print("[启动] 配置信息:")
    print(f"[启动]   - 图片模式：{'本地图片' if USE_LOCAL_IMAGES else '网络图片'}")
    print(f"[启动]   - 测试图片数量：{len(TEST_IMAGES) if not USE_LOCAL_IMAGES else len(LOCAL_IMAGES)}")
    print(f"[启动]   - 模型保存路径：{PROFILE_SAVE_PATH}")
    print(f"[启动]   - 计算设备：{DEVICE}")
    print(f"[启动]   - 调试模式：True")
    print(f"[启动]   - 端口：5000")
    
    print("[启动] 启动 Flask 服务...")
    print("[启动] 服务地址: http://127.0.0.1:5000")
    print("[启动] ==================== 系统启动完成 ====================")
    app.run(debug=True, port=5000)
