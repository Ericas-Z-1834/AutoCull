# AutoCull

面向新闻制作流程的自动化选片工具

## 系统架构

```
AutoCull/
├── main.py              # 主程序 (Web 服务器)
├── index.html           # 选片 Web UI
├── __cull__/            # 选片算法插件目录
│   ├── _.ac.py          # 内置基础选片算法
│   ├── useraes.ac.py    # 用户审美评分算法
│   └── Reg.txt          # 插件开发规范
└── useraes/             # 用户审美建模子系统
    ├── app.py           # Flask 后端 (审美训练)
    └── templates/
        └── index.html   # 审美测试前端
```

## 快速开始

### 1. 安装依赖

```bash
pip install pillow opencv-python numpy torch torchvision flask requests
```

### 2. 启动选片系统

```bash
python main.py
```

访问 http://127.0.0.1:6789

### 3. 训练个人审美模型 (可选)

```bash
cd useraes
python app.py
```

访问 http://127.0.0.1:5000，对图片评分后提交，系统将生成 `user_profile.pkl`。

## 功能说明

### 选片系统 (main.py)

- **图片导入**：支持批量导入本地图片
- **预设选片**：提供多种场景预设（室内会议、室外活动、体育比赛等）
- **评分展示**：显示每张图片的总评分及分项评分
- **半自动模式**：可选择只评分不自动删除
- **图片导出**：将保留的图片导出到 `output/` 目录

### 内置选片算法 (`__cull__/_.ac.py`)

| 算法 | 功能 |
|------|------|
| `builtin_lum` | 亮度评估，检测过曝/欠曝 |
| `builtin_focus` | 对焦质量评估 |
| `builtin_noise` | 噪点水平检测 |
| `builtin_color` | 色彩多样性评估 |

### 用户审美系统 (useraes/)

基于 ResNet18 深度学习模型：
1. 提取图片 512 维特征向量
2. 通过用户评分加权计算个人审美向量
3. 选片时计算余弦相似度进行评分

## 选片预设

系统提供 18 种预设，覆盖常见摄影场景：

| 类别 | 预设 |
|------|------|
| 单算法 | 仅曝光、仅噪点、仅对焦、仅色彩、仅 useraes |
| 室内 | 室内会议_快速/精细 |
| 室外 | 室外活动_运动强/弱_快速/精细 |
| 舞台 | 文艺演出_舞台_快速/精细、体育比赛_快速/精细 |
| 典礼 | 毕业典礼_快速/精细 |
| 校园 | 校园风景、校园建筑_快速/精细 |
| 其他 | 社团活动、讲座、学术报告、采访_快速/精细 |

## 开发选片插件

参考 `__cull__/Reg.txt` 规范：

1. 文件名格式：`作者_功能.ac.py`
2. 函数规范：
```python
def author_func(image: Image.Image = None) -> float | str:
    if not image: return "函数功能描述"
    # 返回 0~1 的评分，0 最差，1 最优
    return score
```

## 目录结构

- `output/` - 导出图片存放目录
- `__pycache__/` - Python 缓存
- `useraes/user_profile.pkl` - 用户审美模型文件
- `config.json` - 配置文件（路径设置）

## 配置文件

项目根目录下的 `config.json` 用于管理个人路径配置：

```json
{
    "useraes": {
        "profile_path": "useraes/user_profile.pkl",
        "local_images_dir": "C:\\Users\\SESIS\\Desktop\\PROJ\\useraes_test\\"
    }
}
```

| 字段 | 说明 |
|------|------|
| `profile_path` | 用户审美模型保存/读取路径 |
| `local_images_dir` | 审美训练用的本地图片目录（留空则使用网络图片） |

## 技术栈

- Python 3
- PIL/Pillow (图像处理)
- OpenCV (边缘检测)
- PyTorch (深度学习特征提取)
- Flask (用户审美后端)
- 原生 HTML/JS (Web UI)
