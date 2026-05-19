"""Basic culling functions for AutoCull - EricasZ"""
import cv2
import numpy as np
from PIL import Image


def builtin_zero(image: Image.Image = None) -> float | str:
    if not image: return "不进行计算而永远返回0的函数，可作占位等用途。"
    return 1.0


def builtin_one(image: Image.Image = None) -> float | str:
    if not image: return "不进行计算而永远返回1的函数，可作占位等用途。"
    return 1.0


def builtin_lum(image: Image.Image = None) -> float | str:
    if not image: return "计算亮度数据，以评估是否过曝或欠曝。"
    IDEAL_LOW = 30      # 理想亮度范围下限
    IDEAL_HIGH = 225    # 理想亮度范围上限
    PENALTY_FACTOR = 0.1  # 惩罚系数，控制评分严格程度

    gray = image.convert('L')
    img_array = np.array(gray)
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))  # 计算直方图
    total_pixels = img_array.size
    dark_pixels = np.sum(hist[:IDEAL_LOW]) / total_pixels      # 过暗像素比例
    bright_pixels = np.sum(hist[IDEAL_HIGH:]) / total_pixels   # 过亮像素比例
    ideal_pixels = np.sum(hist[IDEAL_LOW:IDEAL_HIGH]) / total_pixels  # 理想亮度像素比例
    score = ideal_pixels - (dark_pixels + bright_pixels) * PENALTY_FACTOR  # 计算加权得分，主要基于理想亮度区域的比例，但对过暗和过亮区域进行适度惩罚
    return max(0.0, min(1.0, score))  # 确保得分在范围内


def builtin_focus(image: Image.Image = None) -> float | str:
    if not image: return "评估图像对焦质量，检测图像是否准确且合理对焦。"
    ROI_RATIO = 0.7  # 中心区域占比，用于检测对焦区域
    LOW_THRESHOLD = 30   # Canny边缘检测低阈值
    HIGH_THRESHOLD = 140  # Canny边缘检测高阈值

    img_array = np.array(image.convert('L'))
    edges = cv2.Canny(img_array, LOW_THRESHOLD, HIGH_THRESHOLD)  # 使用Canny边缘检测算法检测边缘
    height, width = edges.shape  # 获取图像尺寸
    total_edge_pixels = np.sum(edges > 0)
    if total_edge_pixels == 0: return 0.0  # 如果没有边缘像素，返回低分
    center_x, center_y = width // 2, height // 2   # 计算中心区域的坐标
    roi_width, roi_height = int(width * ROI_RATIO), int(height * ROI_RATIO)
    roi_edges = edges[
                center_y - roi_height // 2:center_y + roi_height // 2,
                center_x - roi_width // 2:center_x + roi_width // 2]  # 提取中心区域
    roi_edge_pixels = np.sum(roi_edges > 0)  # 计算中心区域内的边缘像素数量
    return roi_edge_pixels / total_edge_pixels if total_edge_pixels > 0 else 0  # 计算中心区域边缘像素占总边缘像素的比例


def builtin_noise(image: Image.Image = None) -> float | str:
    if not image: return "评估图像噪点水平，检测图像是否噪点过多。"
    NOISE_THRESHOLD = 200  # 降低噪点阈值，像素差异超过此值可能为噪点
    LOW_LIGHT_THRESHOLD = 50  # 低亮度阈值，低于此亮度的区域噪点对观感影响较小
    HIGH_NOISE_RATIO = 0.8  # 提高高噪点比例阈值，超过此比例认为噪点过多
    LOW_NOISE_RATIO = 0.02  # 调整低噪点比例阈值，低于此比例认为噪点较少

    gray = image.convert('L')
    img_array = np.array(gray)
    # ---------- 计算相邻像素差异来检测噪点 ----------
    diff_h = np.abs(img_array[1:, :] - img_array[:-1, :])  # 水平方向差值4
    diff_v = np.abs(img_array[:, 1:] - img_array[:, :-1])  # 垂直方向差值
    noise_pixels_h = np.sum(diff_h > NOISE_THRESHOLD)
    noise_pixels_v = np.sum(diff_v > NOISE_THRESHOLD)  # 统计超过阈值的像素差异
    total_comparisons = diff_h.size + diff_v.size
    noise_ratio = (noise_pixels_h + noise_pixels_v) / total_comparisons if total_comparisons > 0 else 0  # 计算噪点比例
    if noise_ratio < LOW_NOISE_RATIO: return 1.0  # 如果噪点很少
    if noise_ratio > HIGH_NOISE_RATIO: return 0.0  # 如果噪点过多
    # ---------- 考虑亮度因素：低亮度区域的噪点对观感影响较小。计算低亮度区域的比例 ----------
    low_light_pixels = np.sum(img_array < LOW_LIGHT_THRESHOLD)
    total_pixels = img_array.size
    low_light_ratio = low_light_pixels / total_pixels if total_pixels > 0 else 0
    adjusted_noise_ratio = noise_ratio * (1 - low_light_ratio * 0.3)  # 在低亮度区域，噪点对观感影响较小，因此调整噪点比例
    normalized_score = 1 - (adjusted_noise_ratio - LOW_NOISE_RATIO) / (HIGH_NOISE_RATIO - LOW_NOISE_RATIO)  # 根据调整后的噪点比例计算得分
    return max(0.0, min(1.0, normalized_score))


def builtin_color(image: Image.Image = None) -> float | str:
    if not image: return "评估图像色彩多样性，检测图像是否色彩过于单一。"
    HUE_BINS = 18         # 色调分桶数量（每20度一个桶）
    MIN_SATURATION = 30   # 最小饱和度阈值，过滤灰度区域
    MIN_VALUE = 40        # 最小亮度阈值，过滤过暗区域
    UNIFORMITY_WEIGHT = 0.6  # 均匀度权重
    DIVERSITY_WEIGHT = 0.4   # 多样性权重

    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image)
    h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]
    
    # 创建有效像素掩码：饱和度和亮度都超过阈值的像素
    valid_mask = (s >= MIN_SATURATION) & (v >= MIN_VALUE)
    valid_hues = h[valid_mask]
    
    if len(valid_hues) == 0:
        return 0.0  # 没有彩色像素，返回最低分
    
    # 计算色调分布直方图
    hue_hist, _ = np.histogram(valid_hues, bins=HUE_BINS, range=(0, 180))
    total_valid = len(valid_hues)
    hue_ratios = hue_hist / total_valid
    
    # 指标1：色调分布均匀度（越均匀得分越高）
    ideal_ratio = 1.0 / HUE_BINS
    uniformity_score = 1.0 - np.sum(np.abs(hue_ratios - ideal_ratio)) / 2.0
    
    # 指标2：色彩多样性（有效彩色像素占比）
    color_diversity = min(1.0, total_valid / (image.width * image.height * 0.3))
    
    # 综合评分
    final_score = UNIFORMITY_WEIGHT * uniformity_score + DIVERSITY_WEIGHT * color_diversity
    return max(0.0, min(1.0, final_score))


# End of _.ac.py - written by EricasZ
