import torch
import clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ─── 設備 ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備: {device}")

# ─── 載入 pytorch-grad-cam ────────────────────────────────
GRAD_CAM_AVAILABLE = False
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    print("✓ 已載入 pytorch_grad_cam")
    GRAD_CAM_AVAILABLE = True
except ImportError:
    print("✗ 未找到 pytorch_grad_cam，請先執行：  pip install grad-cam")
    # 你可以稍後自行補上手動熱圖 fallback，這裡先保持簡單

# ─── 載入 CLIP ────────────────────────────────────────────────
print("載入 CLIP ViT-L/14@336px ...")
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model = model.float()
model.eval()

# ─── 多提示詞對比設定（關鍵改進） ────────────────────────────────
text_descriptions = [
    "a photo of a gray tabby kitten with blue eyes",     # 目標類別 0 ← 我們要高亮的
    "a photo of a golden retriever puppy",               # 1
    "a photo of a dog",
    "a photo of flowers and grass background"            # 背景 3
]

print("\n對比提示詞：")
for i, txt in enumerate(text_descriptions):
    print(f"  {i}: {txt}")

text_tokens = clip.tokenize(text_descriptions).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 包裝模型 → 輸出多類別 logit
class CLIPMultiTextWrapper(torch.nn.Module):
    def __init__(self, clip_model, text_feats):
        super().__init__()
        self.clip_model = clip_model
        self.text_feats = text_feats

    def forward(self, images):
        img_feats = self.clip_model.encode_image(images)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        logits = img_feats @ self.text_feats.T          # [B, num_texts]
        return logits * 4.0                             # temperature scaling (4~100 常見)

clip_wrapper = CLIPMultiTextWrapper(model, text_features).to(device)
clip_wrapper.eval()

# ─── 載入圖片 ────────────────────────────────────────────────
image_path = None
possible_names = ["better_dog_cat.jpg", "dog_cat.jpg", "cat_dog.jpg", "image.jpg", "test.jpg"]
for name in possible_names + [f"./{n}" for n in possible_names]:
    if os.path.exists(name):
        image_path = name
        break

if not image_path:
    image_path = input("找不到圖片，請輸入完整路徑： ").strip()

raw_image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

# 顯示用圖片（保持比例）
disp_size = (336, 336)
disp_img = raw_image.copy()
disp_img.thumbnail(disp_size, Image.LANCZOS)
rgb_img = np.array(disp_img) / 255.0

# ─── 計算機率（debug） ────────────────────────────────
with torch.no_grad():
    probs = clip_wrapper(input_tensor).softmax(dim=-1)[0].cpu().numpy()

print("\n各類別機率：")
for txt, p in zip(text_descriptions, probs):
    print(f"  {txt:50} → {p:.4f}")

# ─── 選擇目標層（修正重點） ────────────────────────────────
# 正確名稱是 ln_1 / ln_2 / attn （不是 norm1）

# 推薦組合（從好到普通）：
target_layers = [model.visual.transformer.resblocks[-1].attn]          # ★ 通常最好

# 其他常見好選擇（可替換上面這行試試）：
# target_layers = [model.visual.transformer.resblocks[-1].ln_2]
# target_layers = [model.visual.transformer.resblocks[-1].ln_1]
# 多層（LayerCAM / EigenCAM 很適合）：
# target_layers = [model.visual.transformer.resblocks[i].ln_2 for i in range(-4, 0)]

print(f"\n使用目標層： {target_layers[0].__class__.__name__} "
      f"(位於 resblocks[-1])")

# ─── 生成 CAM ────────────────────────────────────────────────
if GRAD_CAM_AVAILABLE:
    methods = [
        ("EigenCAM",   EigenCAM,               None),                    # 無需 target
        ("LayerCAM",   LayerCAM,   ClassifierOutputTarget(0)),
        ("GradCAM",    GradCAM,    ClassifierOutputTarget(0)),
        ("GradCAM++",  GradCAMPlusPlus, ClassifierOutputTarget(0)),
    ]
else:
    methods = []

visualizations = []
method_names = ["原始圖"]

for name, MethodClass, target in methods:
    try:
        print(f"生成 {name} ...", end=" ")
        cam_method = MethodClass(model=clip_wrapper,
                                target_layers=target_layers,
                                use_cuda=torch.cuda.is_available())

        if target is None:  # EigenCAM
            grayscale_cam = cam_method(input_tensor=input_tensor,
                                      eigen_smooth=True,
                                      aug_smooth=True)[0, :]
        else:
            grayscale_cam = cam_method(input_tensor=input_tensor,
                                      targets=[target],
                                      eigen_smooth=True,
                                      aug_smooth=True)[0, :]

        # resize & normalize
        h, w = rgb_img.shape[:2]
        grayscale_cam = cv2.resize(grayscale_cam, (w, h))
        grayscale_cam = np.maximum(grayscale_cam, 0)
        if grayscale_cam.max() > 0:
            grayscale_cam /= grayscale_cam.max()

        vis = show_cam_on_image(rgb_img, grayscale_cam,
                               use_rgb=True,
                               colormap=cv2.COLORMAP_JET,
                               image_weight=0.4)
        visualizations.append(vis)
        method_names.append(name)
        print("成功")

    except Exception as e:
        print(f"失敗：{str(e)[:80]}...")

# ─── 畫圖 ────────────────────────────────────────────────
n_vis = len(visualizations)
cols = min(5, n_vis + 1)
rows = (n_vis + 1 + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.5 * rows))
axes = np.atleast_2d(axes).flatten()

axes[0].imshow(rgb_img)
axes[0].set_title("原始圖", fontsize=12)
axes[0].axis('off')

for i, vis in enumerate(visualizations, 1):
    axes[i].imshow(vis)
    axes[i].set_title(method_names[i], fontsize=12)
    axes[i].axis('off')

for ax in axes[n_vis + 1:]:
    ax.axis('off')

plt.suptitle(
    f"CLIP ViT-L/14@336px   —   目標：灰色小貓\n"
    f"機率：{probs[0]:.3f}    (puppy: {probs[1]:.3f})",
    fontsize=14, y=0.96
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("clip_cam_multi_prompt_fixed2.png", dpi=180, bbox_inches='tight')
plt.show()

print("\n完成！結果已存為： clip_cam_multi_prompt_fixed2.png")
print("建議優先觀察 EigenCAM 和 LayerCAM 的熱圖，通常最清晰。")