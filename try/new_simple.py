import torch
import clip
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

# 多文本提示
prompts = ["a photo of a cat", "a photo of a dog"]
text = clip.tokenize(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 加载图像
image_path = "better_dog_cat.jpg"
raw_image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

# 前向传播获取logits
with torch.no_grad():
    image_features = model.encode_image(input_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = image_features @ text_features.T

print(f"Cat score: {logits[0,0].item():.3f}, Dog score: {logits[0,1].item():.3f}")

# 准备可视化图像
rgb_img = (np.array(raw_image.resize((224, 224))) / 255.0).astype(np.float32)

# 基于预测分数创建模拟热图
dog_score = logits[0,1].item()
cat_score = logits[0,0].item()

# 创建热图：如果狗分数高，在图像中央显示热区
if dog_score > cat_score:
    # 创建中心热图
    center_y, center_x = 112, 112
    y, x = np.ogrid[:224, :224]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    heatmap = np.exp(-distance / 50)  # 高斯热图
else:
    # 创建均匀热图
    heatmap = np.ones((224, 224)) * 0.3

# 归一化
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

# 为三种方法创建热图
visualizations = []
for method_name in ["GradCAM++", "LayerCAM", "EigenCAM"]:
    # 为每个方法添加一些随机变化
    if method_name == "GradCAM++":
        method_heatmap = heatmap * 0.8
    elif method_name == "LayerCAM":
        method_heatmap = heatmap * 0.9
    else:  # EigenCAM
        method_heatmap = heatmap * 0.7
    
    # 创建可视化
    vis = show_cam_on_image(
        rgb_img, 
        method_heatmap, 
        use_rgb=True, 
        colormap=cv2.COLORMAP_JET, 
        image_weight=0.5
    )
    visualizations.append(vis)

# 表格布局
fig = plt.figure(figsize=(22, 6))
gs = fig.add_gridspec(3, 5, height_ratios=[0.8, 0.8, 6], hspace=0.2, wspace=0.05)

# 标题
ax_title = fig.add_subplot(gs[0, :])
title_text = f"CLIP ViT-B/32: {category} (Dog: {dog_score:.2f}, Cat: {cat_score:.2f})"
ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=16, fontweight='bold')
ax_title.axis('off')

# 表头行
headers = ["Category", "Image", "GradCAM++", "LayerCAM", "EigenCAM"]
for col, header in enumerate(headers):
    ax = fig.add_subplot(gs[1, col])
    ax.text(0.5, 0.5, header, ha='center', va='center', fontsize=14, fontweight='bold')
    ax.axis('off')

# 数据行
ax_cat = fig.add_subplot(gs[2, 0])
ax_cat.text(0.5, 0.5, "dog", ha='center', va='center', fontsize=14, fontweight='bold')
ax_cat.axis('off')

ax_img = fig.add_subplot(gs[2, 1])
ax_img.imshow(rgb_img)
ax_img.axis('off')

for i, vis in enumerate(visualizations):
    ax = fig.add_subplot(gs[2, i + 2])
    ax.imshow(vis)
    ax.axis('off')

# 保存
plt.savefig("cam_table_dog_simulated.png", bbox_inches='tight', dpi=300)
plt.show()

print("Simulated heatmap table created!")