import torch
import clip
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys
from types import SimpleNamespace

# Grad-CAM imports
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings('ignore')

# ==========================
# 基础设置
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ==========================
# 1. 查找图像
# ==========================
image_paths_to_try = [
    "better_dog_cat.jpg",
    "./better_dog_cat.jpg",
    "../better_dog_cat.jpg",
    "dog_cat.jpg",
    "cat_dog.jpg",
    "test.jpg",
    "image.jpg",
    "example.jpg"
]

image_path = None
for path in image_paths_to_try:
    if os.path.exists(path):
        image_path = path
        print(f"找到图像文件: {path}")
        break

if image_path is None:
    print("错误: 没有找到图像文件。请提供图像路径:")
    image_path = input("请输入图像文件路径: ")

# ==========================
# 2. 加载 CLIP 模型
# ==========================
print("加载 CLIP 模型 (ViT-L/14@336px)...")
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model = model.to(device).float()
model.eval()
try:
    first_param_dtype = next(model.parameters()).dtype
except StopIteration:
    first_param_dtype = torch.float32
print("模型 dtype:", first_param_dtype)

# ==========================
# 3. 包装模型的视觉编码器以确保正确的张量形状
# ==========================
class CLIPVisualWrapper(torch.nn.Module):
    """
    Wrapper around CLIP visual encoder that ensures output shape is [B, N, C]
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
    
    def forward(self, x):
        """
        x: [B, 3, H, W]
        Returns: [B, N, C] where N = num_patches + 1 (class token)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.visual.conv1(x)  # [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, N_patches]
        x = x.permute(0, 2, 1)  # [B, N_patches, C]
        
        # Add class token
        cls_token = self.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N_patches, C]
        
        # Add positional embeddings
        x = x + self.visual.positional_embedding.unsqueeze(0).to(x.dtype)  # [B, N, C]
        
        # Pass through transformer
        x = self.visual.transformer(x)  # [B, N, C]
        
        return x  # Ensure [B, N, C] format


class CLIPCatVsDogModelV2(torch.nn.Module):
    def __init__(self, visual_wrapper, cat_text_feat, dog_text_feat):
        super().__init__()
        self.visual = visual_wrapper
        self.register_buffer("cat_text", cat_text_feat.detach().clone().float())
        self.register_buffer("dog_text", dog_text_feat.detach().clone().float())

    def forward(self, x):
        # x: [B, 3, H, W]
        # visual forward: [B, N, C]
        x = self.visual(x)
        
        # Pool using class token (first token)
        img = x[:, 0, :]  # [B, C]
        img = img / img.norm(dim=-1, keepdim=True)
        
        cat = self.cat_text / self.cat_text.norm(dim=-1, keepdim=True)
        dog = self.dog_text / self.dog_text.norm(dim=-1, keepdim=True)
        
        s_cat = img @ cat.T  # [B, 1]
        s_dog = img @ dog.T  # [B, 1]
        
        logits = s_cat - s_dog  # [B, 1]
        return logits

# ==========================
# 4. 文本提示词与特征
# ==========================
prompts_to_try = [
    "a photo of a cat",
    "a gray cat",
    "a kitten",
    "a cat face",
    "a domestic cat",
    "a gray kitten with blue eyes",
    "a close up photo of a cat",
    "a cat looking at camera",
    "a cute cat"
]

print("\n尝试不同提示词:")
for i, p in enumerate(prompts_to_try):
    print(f" {i+1}. {p}")

selected_prompt_idx = 8
prompt = prompts_to_try[selected_prompt_idx]
category = "Cat"

print(f"\n选择的提示词: '{prompt}'")
print(f"类别: {category}")

text = clip.tokenize([prompt]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text).float()

cat_text = clip.tokenize(["a cute cat"]).to(device)
dog_text = clip.tokenize(["a cute dog"]).to(device)
with torch.no_grad():
    cat_text_feat = model.encode_text(cat_text).float()
    dog_text_feat = model.encode_text(dog_text).float()

# ==========================
# 5. 加载与预处理图像
# ==========================
print(f"\n加载图像: {image_path}")
raw_image = Image.open(image_path).convert("RGB")
print(f"原始图像尺寸: {raw_image.size}")

input_tensor = preprocess(raw_image).unsqueeze(0).to(device).float()
input_tensor.requires_grad_(True)

vis_size = 336
vis_image = raw_image.resize((vis_size, vis_size))
rgb_img = np.array(vis_image).astype(np.float32) / 255.0
print(f"可视化图像尺寸: {rgb_img.shape}")

with torch.no_grad():
    image_features = model.encode_image(input_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features_norm.T).item()

print(f"模型对提示词 '{prompt}' 的相似度: {similarity:.4f}")

# ==========================
# 6. 创建包装模型
# ==========================
visual_wrapper = CLIPVisualWrapper(model).to(device).float()
cam_model = CLIPCatVsDogModelV2(visual_wrapper, cat_text_feat, dog_text_feat).to(device).float()
cam_model.eval()

# ==========================
# 7. 验证输出形状
# ==========================
print("\n验证输出形状...")
with torch.no_grad():
    test_out = visual_wrapper(input_tensor)
    print(f"Visual wrapper output shape: {test_out.shape}")
    if test_out.shape[1] <= 1:
        print("ERROR: Still getting incorrect shape!")
        sys.exit(1)

# ==========================
# 8. 选择目标层
# ==========================
# Target the transformer blocks or attention heads
target_layer = visual_wrapper.visual.transformer.resblocks[-2]
target_layers = [target_layer]

print(f"目标层: visual.transformer.resblocks[-2]")

# ==========================
# 9. reshape_transform
# ==========================
def reshape_transform(tensor):
    """
    Transform [B, N, C] to [B, C, H, W]
    """
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, N, C], got shape {tensor.shape}")
    
    B, N, C = tensor.shape
    print(f"reshape_transform input: B={B}, N={N}, C={C}")
    
    if N <= 1:
        raise ValueError(f"Token count N={N} too small for CAM (need N > 1). "
                        f"Tensor shape: {tensor.shape}")
    
    num_patches = N - 1  # Exclude class token
    
    # Find valid side length
    side = int(round(num_patches ** 0.5))
    
    if side * side != num_patches:
        # For ViT-L/14@336px, patch_size=14, so side = 336/14 = 24
        patch_size = 14
        side = 336 // patch_size
        if side * side != num_patches:
            print(f"Warning: num_patches={num_patches}, side={side}, side²={side*side}")
    
    print(f"Reshaping to side={side}")
    
    # Reshape: skip class token
    result = tensor[:, 1:, :].reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()
    return result

# ==========================
# 10. 生成 CAM 热图
# ==========================
print("\n生成热图...")

targets = [ClassifierOutputTarget(0)]
visualizations = []
method_names = []
cam_methods_to_try = []

# Initialize CAM methods
try:
    cam_methods_to_try.append(("GradCAM", GradCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform)))
except Exception as e:
    print(f"GradCAM 初始化失败: {e}")

try:
    cam_methods_to_try.append(("GradCAM++", GradCAMPlusPlus(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform)))
except Exception as e:
    print(f"GradCAM++ 初始化失败: {e}")

try:
    cam_methods_to_try.append(("LayerCAM", LayerCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform)))
except Exception as e:
    print(f"LayerCAM 初始化失败: {e}")

try:
    cam_methods_to_try.append(("EigenCAM", EigenCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform)))
except Exception as e:
    print(f"EigenCAM 初始化失败: {e}")

for name, cam_method in cam_methods_to_try:
    try:
        print(f" 正在生成 {name}...")
        if name == "EigenCAM":
            grayscale_cam = cam_method(input_tensor=input_tensor)[0, :]
        else:
            with torch.enable_grad():
                grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)[0, :]

        if np.max(grayscale_cam) > 0:
            grayscale_cam = grayscale_cam / np.max(grayscale_cam)

        if grayscale_cam.shape != rgb_img.shape[:2]:
            grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5)
        visualizations.append(vis)
        method_names.append(name)
        print(f" ✓ {name} 成功！")

    except Exception as e:
        print(f" ✗ 生成 {name} 时出错: {e}")
        import traceback
        traceback.print_exc()

if len(visualizations) == 0:
    print("\n错误: 所有 CAM 方法均失败。")
    sys.exit(1)

# ==========================
# 11. 绘制对比表格
# ==========================
print("\n创建可视化表格...")
n_methods = len(visualizations)
n_cols = 2 + n_methods

fig = plt.figure(figsize=(5 * n_cols, 8))
gs = fig.add_gridspec(3, n_cols, height_ratios=[0.8, 0.8, 6], hspace=0.2, wspace=0.05)

ax_title = fig.add_subplot(gs[0, :])
title_text = f"CLIP ViT-L/14@336px - 聚焦: '{prompt}' (相似度: {similarity:.4f})"
ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=16, fontweight='bold')
ax_title.axis('off')

headers = ["类别", "原始图像"] + method_names
for col, header in enumerate(headers):
    ax = fig.add_subplot(gs[1, col])
    ax.text(0.5, 0.5, header, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.axis('off')

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

plt.tight_layout()
output_path = "gradcam_final_results.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"\n✓ 热图生成完成！")
print(f"✓ 保存到: {output_path}")
print(f"✓ 使用的提示词: '{prompt}'")
print(f"✓ 相似度: {similarity:.4f}")
print(f"✓ 成功生成的热图方法: {', '.join(method_names)}")
print("\n✓ 所有任务完成！")