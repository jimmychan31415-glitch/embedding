import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

def get_patch_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=True
        )
        hidden_states = vision_outputs.last_hidden_state
        patches = hidden_states[:, 1:, :]  # [1, num_patches, 1024]
    
    num_patches = patches.shape[1]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        print(f"警告：{image_path} patch 數 {num_patches} 不是完美平方")
    patches = patches.reshape(1, grid_size, grid_size, -1)
    print(f"{image_path} patch grid: {grid_size}x{grid_size}")
    return patches, image.size

def compute_text_guided_heatmap(patches, text_prompt):
    text_inputs = processor(text=[text_prompt], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)  # [1, projected_dim]

    # 投影 patches 到對齊空間
    patches_flat = patches.view(-1, patches.shape[-1])  # [num_patches, 1024]
    projected_patches = model.visual_projection(patches_flat)  # [num_patches, projected_dim]

    # 正規化
    projected_patches_norm = F.normalize(projected_patches, p=2, dim=-1)
    text_norm = F.normalize(text_features, p=2, dim=-1)

    # 相似度計算
    sim = torch.matmul(projected_patches_norm, text_norm.T)  # [num_patches, 1]

    # 斷開梯度 + 轉 numpy（關鍵修正！）
    sim = sim.detach().squeeze(1).cpu().numpy()

    # 重塑回 grid
    H, W = patches.shape[1], patches.shape[2]
    heatmap = sim.reshape(H, W)
    return heatmap

def compute_text_guided_heatmap(patches, text_prompt):
    with torch.no_grad():
        text_inputs = processor(text=[text_prompt], return_tensors="pt", padding=True)
        text_features = model.get_text_features(**text_inputs)

        patches_flat = patches.view(-1, patches.shape[-1])
        projected_patches = model.visual_projection(patches_flat)

        projected_patches_norm = F.normalize(projected_patches, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)

        sim = torch.matmul(projected_patches_norm, text_norm.T)
        sim = sim.detach().squeeze(1).cpu().numpy()

        H, W = patches.shape[1], patches.shape[2]
        heatmap = sim.reshape(H, W)
    return heatmap

def visualize_heatmap(original_image_path, heatmap, text_prompt, output_file='clip_text_guided_heatmap.png'):
    image = Image.open(original_image_path).convert("RGB")
    orig_w, orig_h = image.size
    
    heatmap_resized = zoom(heatmap, (orig_h / heatmap.shape[0], orig_w / heatmap.shape[1]))
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.imshow(heatmap_resized, cmap='hot', alpha=0.7)
    plt.title(f"文字引導熱圖：'{text_prompt}'\n(紅色 = 高相關區域)")
    plt.axis("off")
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"文字引導熱圖已儲存為 {output_file}")
    plt.close()

if __name__ == "__main__":
    query_path = "query_image.jpg"
    text_prompt = "iron lattice tower structure"
    
    print("開始處理圖片...")
    patches, _ = get_patch_embeddings(query_path)
    
    print(f"使用文字引導：'{text_prompt}'")
    heatmap = compute_text_guided_heatmap(patches, text_prompt)
    
    print("生成並儲存視覺化結果...")
    visualize_heatmap(query_path, heatmap, text_prompt)
    print("完成！請下載 clip_text_guided_heatmap.png 查看結果。")