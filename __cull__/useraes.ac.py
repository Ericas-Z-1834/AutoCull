"""Culling function based on useraes - EricasZ"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ================= 配置常量（便于调试） =================
import json
import os

# 加载外部配置
_config = {}
_config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
if os.path.exists(_config_path):
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config = json.load(f)

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用户审美文件路径（从配置文件读取）
DEFAULT_PROFILE_PATH = _config.get('useraes', {}).get('profile_path', "useraes/user_profile.pkl")

# 图像预处理参数（必须与 app.py 保持一致）
RESIZE_SIZE = 256             # 图像缩放尺寸
CROP_SIZE = 224               # 中心裁剪尺寸
NORMALIZE_MEAN = [0.485, 0.456, 0.406]   # 归一化均值
NORMALIZE_STD = [0.229, 0.224, 0.225]    # 归一化标准差

# 模型配置
MODEL_NAME = "resnet18"     # 使用的模型名称
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"  # 预训练权重版本
FEATURE_POOL_SIZE = (1, 1)  # 特征池化尺寸

# 评分映射参数
SIMILARITY_OFFSET = -0.5    # 余弦相似度偏移值
SIMILARITY_SCALE = 1.0      # 余弦相似度缩放值

# ================= 全局状态（单例模式） =================
_model = None
_feature_extractor = None
_preprocess = None


def ericasz_useraes(image_input: Image.Image = None, profile_path=DEFAULT_PROFILE_PATH):
    """基于用户审美量化数据，计算图片的用户审美评分。
    
    Args:
        image_input: PIL.Image 对象或图片路径
        profile_path: 用户审美文件路径
        
    Returns:
        float: 审美评分 [0, 1]
    """
    global _model, _feature_extractor, _preprocess
    
    if not image_input:
        return "基于用户审美量化数据，计算图片的用户审美评分。"
    
    # ========== 1. 初始化模型和预处理（单例模式） ==========
    if _feature_extractor is None or _preprocess is None:
        print("加载评分模型...")
        # 初始化特征提取器
        weights = getattr(models, MODEL_NAME.replace('_', '').capitalize() + '_Weights', None)
        if weights is not None:
            pretrained_weights = getattr(weights, PRETRAINED_WEIGHTS)
        else:
            pretrained_weights = True  # 降级使用布尔值
        _model = models.__dict__[MODEL_NAME](weights=pretrained_weights)
        modules = list(_model.children())[:-2]
        _feature_extractor = nn.Sequential(*modules)
        _feature_extractor.to(DEVICE)
        _feature_extractor.eval()
        
        # 初始化预处理
        _preprocess = transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
    
    # ========== 2. 加载用户审美向量 ==========
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"未找到用户审美文件：{profile_path}，请先运行后端测试系统。")
    
    with open(profile_path, 'rb') as f:
        profile_data = pickle.load(f)
    
    user_vector = profile_data['vector']
    
    # ========== 3. 提取输入图片特征 ==========
    # 处理输入图片
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"图片路径不存在：{image_input}")
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError("输入必须是图片路径 (str) 或 PIL.Image 对象")
    
    # 预处理并提取特征
    input_tensor = _preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = _feature_extractor(input_batch)
        features = nn.functional.adaptive_avg_pool2d(features, FEATURE_POOL_SIZE)
        features = features.squeeze()
    
    # 特征向量 L2 归一化
    feature_np = features.cpu().numpy()
    norm = np.linalg.norm(feature_np)
    if norm > 0:
        feature_np /= norm
    
    # ========== 4. 计算余弦相似度并映射到 [0, 1] ==========
    # 由于向量已 L2 归一化，点积即为余弦相似度
    similarity = np.dot(user_vector, feature_np)
    
    # 调试信息
    print(f"[调试] 用户向量范数: {np.linalg.norm(user_vector):.6f}")
    print(f"[调试] 图片特征范数: {np.linalg.norm(feature_np):.6f}")
    print(f"[调试] 余弦相似度: {similarity:.6f}")
    
    # 余弦相似度范围 [-1, 1]，映射到 [0, 1]
    score = (similarity + 1.0) / 2

    # 经测试，返回得分普遍位于 0.8-0.9 间，进行重映射
    normalized = 10 * score - 8.0

    return float(max(0.0, min(1.0, normalized)))

# End of useraes.ac.py - written by EricasZ

# ================= 测试代码 =================


if __name__ == "__main__":
    # 测试图像路径
    test_image_path = r"C:\Users\SESIS\Pictures\ComfyUI\ComfyUI_temp_yqolj_00001_.png"
    
    print("=" * 50)
    print("用户审美评分系统测试")
    print("=" * 50)
    
    try:
        # 调用评分函数
        score = ericasz_useraes(Image.open(test_image_path))
        
        print(f"\n测试图像: {test_image_path}")
        print(f"审美评分: {score:.4f}")
        print(f"评分等级: {'优秀' if score >= 0.8 else '良好' if score >= 0.6 else '一般' if score >= 0.4 else '较差'}")
        print("\n测试完成！")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n未知错误: {type(e).__name__}: {e}")
