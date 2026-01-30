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
# 3. 定义适配 Grad-CAM 的模型
# ==========================
class CLIPSimilarityModel(torch.nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_features", text_features.detach().clone().float())

    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        return logits


class CLIPCatVsDogModel(torch.nn.Module):
    def __init__(self, clip_model, cat_text_feat, dog_text_feat):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("cat_text", cat_text_feat.detach().clone().float())
        self.register_buffer("dog_text", dog_text_feat.detach().clone().float())

    def forward(self, x):
        img = self.clip_model.encode_image(x)
        img = img / img.norm(dim=-1, keepdim=True)
        cat = self.cat_text / self.cat_text.norm(dim=-1, keepdim=True)
        dog = self.dog_text / self.dog_text.norm(dim=-1, keepdim=True)
        s_cat = img @ cat.T
        s_dog = img @ dog.T
        logits = s_cat - s_dog
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
# 6. Wrapper to extract intermediate activations
# ==========================
class ActivationExtractor:
    """Extract intermediate layer activations"""
    def __init__(self):
        self.activation = None
    
    def __call__(self, module, input, output):
        self.activation = output

# ==========================
# 7. 创建适配模型 & 选择目标层
# ==========================
use_cat_vs_dog = True

if use_cat_vs_dog:
    cam_model = CLIPCatVsDogModel(model, cat_text_feat, dog_text_feat).to(device).float()
    cam_title_suffix = " (cat vs dog contrast)"
else:
    cam_model = CLIPSimilarityModel(model, text_features).to(device).float()
    cam_title_suffix = ""

cam_model.eval()

# FIX: Test layers with actual forward pass to find valid target
print("\n寻找合适的目标层...")
visual = cam_model.clip_model.visual
resblocks = getattr(visual.transformer, "resblocks", [])
n_blocks = len(resblocks)

target_layer = None
target_shape = None

# Strategy: Try resblocks in reverse order, looking for ones that give meaningful activations
candidates_to_test = [
    ("transformer.resblocks[-1]", visual.transformer.resblocks[-1]),
    ("transformer.resblocks[-2]", visual.transformer.resblocks[-2]),
    ("transformer.resblocks[-3]", visual.transformer.resblocks[-3]),
]

for name, module in candidates_to_test:
    extractor = ActivationExtractor()
    hook = module.register_forward_hook(extractor)
    
    try:
        with torch.no_grad():
            _ = cam_model(input_tensor)
        
        if extractor.activation is not None:
            act = extractor.activation
            print(f"Layer {name}: output shape = {act.shape}")
            
            # We want [B, N, C] shape
            if isinstance(act, torch.Tensor) and act.dim() == 3:
                B, N, C = act.shape
                if N > 1:
                    target_layer = module
                    target_shape = act.shape
                    print(f"✓ Selected {name} with shape {target_shape}")
                    break
    except Exception as e:
        print(f"Error testing {name}: {e}")
    finally:
        hook.remove()

if target_layer is None:
    print("Warning: Could not find optimal layer. Using resblocks[-1]...")
    target_layer = visual.transformer.resblocks[-1]
    target_shape = None

target_layers = [target_layer]

# ==========================
# 8. Improved reshape_transform
# ==========================
def reshape_transform(tensor):
    """
    Transform tensor from [B, N, C] to [B, C, H, W]
    Handles edge cases more robustly
    """
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")
    
    # Ensure 3D
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, N, C], got shape {tensor.shape}")
    
    B, N, C = tensor.shape
    
    if N <= 1:
        raise ValueError(f"Token count N={N} too small for CAM (need N > 1)")
    
    # Account for class token
    num_patches = N - 1
    
    # Try to find valid side length
    side = int(round(num_patches ** 0.5))
    
    if side * side != num_patches:
        # If not perfect square, try common divisors
        for candidate_side in [24, 26, 20, 16, 14]:
            if candidate_side * candidate_side == num_patches:
                side = candidate_side
                break
        else:
            # Fallback: use detected patch size
            patch_size = 14
            side = int(input_tensor.shape[-1] // patch_size)
            if side * side != num_patches:
                print(f"Warning: num_patches={num_patches} is not a perfect square, forcing side={side}")
    
    # Reshape: skip class token (index 0)
    result = tensor[:, 1:, :].reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()
    return result

# ==========================
# 9. 生成 CAM 热图
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

    except Exception as e:
        print(f" 生成 {name} 时出错: {e}")

if len(visualizations) == 0:
    print("错误: 所有 CAM 方法均失败。")
    print("建议:")
    print("  1. 确保 pytorch-grad-cam 已正确安装: pip install pytorch-grad-cam")
    print("  2. 检查 CLIP 模型的 visual.transformer.resblocks 是否存在")
    print("  3. 尝试手动指定更早的层，如 resblocks[-5] 或 resblocks[-10]")
    sys.exit(1)

# ==========================
# 10. 绘制对比表格
# ==========================
print("\n创建可视化表格...")
n_methods = len(visualizations)
n_cols = 2 + n_methods

fig = plt.figure(figsize=(5 * n_cols, 8))
gs = fig.add_gridspec(3, n_cols, height_ratios=[0.8, 0.8, 6], hspace=0.2, wspace=0.05)

ax_title = fig.add_subplot(gs[0, :])
title_text = f"CLIP ViT-L/14@336px - 聚焦: '{prompt}' (相似度: {similarity:.4f}){cam_title_suffix}"
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
print("\n--- 调试信息 ---")
print(f"输入张量形状: {input_tensor.shape}, dtype: {input_tensor.dtype}, requires_grad: {input_tensor.requires_grad}")
print(f"文本特征形状: {text_features.shape}, dtype: {text_features.dtype}")
print(f"显示图像形状: {rgb_img.shape}")
print(f"模式: {'猫 vs 狗对比' if use_cat_vs_dog else '单提示词'}")
print(f"目标层激活形状: {target_shape}")
print("\n✓ 所有任务完成！")