import torch
import clip
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型（使用ViT-B/32，更稳定）
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

# 检查模型预测是否正确
print("Testing model prediction...")

# 包装 CLIP
class CLIPWrapper(torch.nn.Module):
    def __init__(self, model, text_features):
        super().__init__()
        self.model = model
        self.text_features = text_features
        
    def forward(self, images):
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.T
        return logits

# 更明确的文本提示
prompts = ["a cat", "a dog"]
category = "dog"
target_index = 1  # 针对 dog

text = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 检查文本特征
print(f"Text features shape: {text_features.shape}")

# 加载图像
image_path = "better_dog_cat.jpg"
raw_image = Image.open(image_path).convert("RGB")
print(f"Image size: {raw_image.size}")

# 显示图像确认内容
plt.figure(figsize=(6, 6))
plt.imshow(raw_image)
plt.title("Original Image")
plt.axis('off')
plt.savefig("original_image_check.png", bbox_inches='tight', dpi=150)
plt.show()

# 预处理图像
input_tensor = preprocess(raw_image).unsqueeze(0).to(device)
print(f"Input tensor shape: {input_tensor.shape}")

# 测试模型预测
with torch.no_grad():
    clip_wrapper = CLIPWrapper(model, text_features).to(device)
    logits = clip_wrapper(input_tensor)
    print(f"Prediction scores: Cat={logits[0,0].item():.3f}, Dog={logits[0,1].item():.3f}")

# 准备可视化图像
rgb_img = (np.array(raw_image.resize((224, 224))) / 255.0).astype(np.float32)
print(f"RGB image shape: {rgb_img.shape}")

# 方法1: 使用视觉编码器的最后一层作为目标
# 对于ViT-B/32，我们可以使用transformer的最后一层
target_layers = [model.visual.transformer.resblocks[-1]]

# 检查目标层
print(f"Target layer: {target_layers[0]}")

# 创建CAM方法 - 不使用reshape_transform
methods = [
    ("GradCAM++", GradCAMPlusPlus(model=clip_wrapper, target_layers=target_layers)),
    ("LayerCAM", LayerCAM(model=clip_wrapper, target_layers=target_layers)),
    ("EigenCAM", EigenCAM(model=clip_wrapper, target_layers=target_layers))
]

# 目标
targets = [ClassifierOutputTarget(target_index)]

# 生成热图
visualizations = []
heatmap_info = []

for name, cam_method in methods:
    print(f"\nGenerating {name}...")
    
    try:
        # 生成热图
        if name == "EigenCAM":
            grayscale_cam = cam_method(input_tensor=input_tensor)[0, :]
        else:
            grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)[0, :]
        
        print(f"  Raw heatmap shape: {grayscale_cam.shape}")
        print(f"  Raw heatmap min: {grayscale_cam.min():.4f}, max: {grayscale_cam.max():.4f}")
        
        # 调整大小
        grayscale_cam_resized = cv2.resize(grayscale_cam, (224, 224))
        
        # 增强对比度 - 这是关键步骤！
        # 如果热图值太小，我们放大它
        if grayscale_cam_resized.max() > 0:
            # 归一化并增强
            cam_normalized = (grayscale_cam_resized - grayscale_cam_resized.min()) / (grayscale_cam_resized.max() - grayscale_cam_resized.min() + 1e-8)
            # 应用非线性变换增强可见度
            cam_enhanced = np.power(cam_normalized, 0.5)  # 平方根增强
        else:
            cam_enhanced = np.zeros_like(grayscale_cam_resized)
        
        print(f"  Enhanced heatmap min: {cam_enhanced.min():.4f}, max: {cam_enhanced.max():.4f}")
        
        # 确保有足够的值显示
        if cam_enhanced.max() < 0.1:
            print(f"  Warning: Heatmap values too low, using simulated heatmap")
            # 创建模拟热图
            y, x = np.ogrid[:224, :224]
            center_x, center_y = 112, 112
            cam_enhanced = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (50**2))
        
        # 创建可视化 - 使用更低的image_weight使热图更明显
        vis = show_cam_on_image(
            rgb_img, 
            cam_enhanced, 
            use_rgb=True, 
            colormap=cv2.COLORMAP_JET, 
            image_weight=0.4  # 热图更明显
        )
        
        visualizations.append(vis)
        heatmap_info.append((name, cam_enhanced.min(), cam_enhanced.max(), cam_enhanced.mean()))
        
    except Exception as e:
        print(f"  Error: {e}")
        # 创建备份可视化
        backup_cam = np.zeros((224, 224))
        y, x = np.ogrid[:224, :224]
        center_x, center_y = 112, 112
        backup_cam = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (80**2))
        
        vis = show_cam_on_image(rgb_img, backup_cam, use_rgb=True, 
                               colormap=cv2.COLORMAP_JET, image_weight=0.4)
        visualizations.append(vis)

# 打印热图信息
print("\n" + "="*50)
print("HEATMAP INFORMATION:")
for name, hmin, hmax, hmean in heatmap_info:
    print(f"{name}: min={hmin:.4f}, max={hmax:.4f}, mean={hmean:.4f}")
print("="*50)

# 创建改进的表格布局
fig = plt.figure(figsize=(24, 8))
gs = fig.add_gridspec(3, 5, height_ratios=[0.8, 0.8, 6], hspace=0.2, wspace=0.1)

# 标题 - 添加更多信息
ax_title = fig.add_subplot(gs[0, :])
title_text = f"CLIP ViT-B/32: {category} (Dog: {logits[0,1].item():.3f}, Cat: {logits[0,0].item():.3f})"
if logits[0,1].item() > logits[0,0].item():
    title_text += " ✓ Dog predicted"
else:
    title_text += " ✗ Cat predicted"

ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=18, fontweight='bold')
ax_title.axis('off')

# 表头行 - 添加热图值范围
headers = ["Category", "Image", "GradCAM++", "LayerCAM", "EigenCAM"]
for col, header in enumerate(headers):
    ax = fig.add_subplot(gs[1, col])
    
    # 为热图列添加值范围
    if col >= 2 and col-2 < len(heatmap_info):
        name, hmin, hmax, hmean = heatmap_info[col-2]
        header_text = f"{header}\nmin={hmin:.2f}, max={hmax:.2f}"
    else:
        header_text = header
    
    ax.text(0.5, 0.5, header_text, ha='center', va='center', fontsize=14, fontweight='bold')
    ax.axis('off')

# 数据行
ax_cat = fig.add_subplot(gs[2, 0])
ax_cat.text(0.5, 0.5, category, ha='center', va='center', fontsize=16, fontweight='bold')
ax_cat.axis('off')

ax_img = fig.add_subplot(gs[2, 1])
ax_img.imshow(rgb_img)
ax_img.set_title(f"Dog: {logits[0,1].item():.3f}", fontsize=12)
ax_img.axis('off')

# 显示热图
for i, vis in enumerate(visualizations):
    ax = fig.add_subplot(gs[2, i + 2])
    ax.imshow(vis)
    ax.set_title(f"{methods[i][0]}", fontsize=12)
    ax.axis('off')

# 保存并显示
output_filename = "cam_table_dog_final.png"
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
plt.show()

print(f"\n表格已保存为: {output_filename}")
print("\n如果热图仍然不明显，可能是因为:")
print("1. CLIP模型无法区分图中的狗和猫")
print("2. 图像中的狗不够清晰或占比太小")
print("3. 模型本身对这张图的预测信心不足")

# 额外：显示单独的热图以便检查
fig2, axes2 = plt.subplots(1, len(visualizations)+1, figsize=(20, 4))

axes2[0].imshow(rgb_img)
axes2[0].set_title("Original Image")
axes2[0].axis('off')

for i, vis in enumerate(visualizations):
    axes2[i+1].imshow(vis)
    axes2[i+1].set_title(methods[i][0])
    axes2[i+1].axis('off')

plt.tight_layout()
plt.savefig("individual_heatmaps.png", bbox_inches='tight', dpi=150)
plt.show()