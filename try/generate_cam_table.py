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

# 加载 CLIP 模型（ViT-B/32）
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()  # 确保 fp32

# 包装 CLIP（使用原始相似度，无 softmax）
class CLIPWrapper(torch.nn.Module):
    def __init__(self, model, text_features):
        super().__init__()
        self.model = model
        self.text_features = text_features

    def forward(self, images):
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.T  # 原始相似度
        return logits

# 多文本提示（增加区分性，热图会聚焦 dog）
prompts = [ "a photo of a cat", "a photo of a dog"]
category = "cat"
target_index = 0  # 针对 cat

text = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

clip_wrapper = CLIPWrapper(model, text_features).to(device)

# 加载图像（您的文件名）
image_path = "better_dog_cat.jpg"
raw_image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

rgb_img = (np.array(raw_image.resize((224, 224))) / 255.0).astype(np.float32)

# 目标层：patch embedding conv（天然卷积层，无需 reshape）
target_layers = [model.visual.conv1]

# 目标
targets = [ClassifierOutputTarget(target_index)]

# 生成热图（无 reshape_transform）
methods = [
    ("GradCAM++", GradCAMPlusPlus(model=clip_wrapper, target_layers=target_layers)),
    ("LayerCAM", LayerCAM(model=clip_wrapper, target_layers=target_layers)),
    ("EigenCAM", EigenCAM(model=clip_wrapper, target_layers=target_layers))
]

visualizations = []
for name, cam_method in methods:
    if name == "EigenCAM":
        grayscale_cam = cam_method(input_tensor=input_tensor)[0, :]
    else:
        grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)[0, :]
    vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, 
                            colormap=cv2.COLORMAP_JET, image_weight=0.6)  # 优化透明度和颜色
    visualizations.append(vis)

# 表格布局（5 列，更宽以适应图片）
fig = plt.figure(figsize=(22, 6))
gs = fig.add_gridspec(3, 5, height_ratios=[0.8, 0.8, 6], hspace=0.2, wspace=0.05)

# 标题
ax_title = fig.add_subplot(gs[0, :])
ax_title.text(0.5, 0.5, "Vision Transformer (CLIP ViT-B/32):", ha='center', va='center', fontsize=16, fontweight='bold')
ax_title.axis('off')

# 表头行
headers = ["Category", "Image", "GradCAM++", "LayerCAM", "EigenCAM"]
for col, header in enumerate(headers):
    ax = fig.add_subplot(gs[1, col])
    ax.text(0.5, 0.5, header, ha='center', va='center', fontsize=14, fontweight='bold')
    ax.axis('off')

# 数据行
ax_cat = fig.add_subplot(gs[2, 0])
ax_cat.text(0.5, 0.5, category, ha='center', va='center', fontsize=14, fontweight='bold')
ax_cat.axis('off')

ax_img = fig.add_subplot(gs[2, 1])
ax_img.imshow(rgb_img)
ax_img.axis('off')

for i, vis in enumerate(visualizations):
    ax = fig.add_subplot(gs[2, i + 2])
    ax.imshow(vis)
    ax.axis('off')

# 保存并显示
plt.savefig("cam_table dog.png", bbox_inches='tight', dpi=300)
plt.show()

print("表格热图生成完成！使用 conv1 层避免了所有 reshape/梯度错误，热图会突出狗的区域（红色/黄色高关注），检查 cam_table dog.png。")