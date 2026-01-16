import matplotlib
matplotlib.use('Agg')  # 使用非互動後端，只存檔不顯示

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

# 載入模型
model_name = "facebook/dinov2-large"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def get_patch_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    patches = outputs.last_hidden_state[:, 1:, :]  # [1, num_patches, D]
    num_patches = patches.shape[1]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        print(f"警告：{image_path} patch 數 {num_patches} 不是完美平方，使用近似 grid")
    patches = patches.reshape(1, grid_size, grid_size, -1)
    print(f"{image_path} patch grid: {grid_size}x{grid_size}")
    return patches, image.size

def compute_similarity_heatmap(query_patches, target_patches):
    # 展平空間維度
    query_flat = query_patches.view(1, -1, query_patches.shape[-1])   # (1, num_q, D)
    target_flat = target_patches.view(1, -1, target_patches.shape[-1]) # (1, num_t, D)
    
    # 正規化
    query_norm = F.normalize(query_flat, p=2, dim=-1)
    target_norm = F.normalize(target_flat, p=2, dim=-1)
    
    # 計算相似度矩陣
    sim_matrix = torch.matmul(query_norm, target_norm.transpose(1, 2))  # (1, num_q, num_t)
    max_sim = sim_matrix.max(dim=-1)[0]  # (1, num_q)
    
    # 重塑回 query 的空間格子
    H_q, W_q = query_patches.shape[1], query_patches.shape[2]
    heatmap = max_sim.view(H_q, W_q).cpu().numpy()
    return heatmap

def visualize_heatmap(original_image_path, heatmap, output_file='heatmap_result.png'):
    image = Image.open(original_image_path).convert("RGB")
    orig_w, orig_h = image.size
    
    # 上採樣熱圖
    heatmap_resized = zoom(heatmap, (orig_h / heatmap.shape[0], orig_w / heatmap.shape[1]))
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # 繪圖
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("原圖 (Query)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap_resized, cmap='hot', alpha=0.7)
    plt.title("相似度熱圖（紅色 = 高匹配）")
    plt.axis("off")
    
    # 儲存檔案
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"熱圖已儲存為 {output_file}")
    plt.close()

# 主程式
if __name__ == "__main__":
    query_path = "query_image.jpg"
    target_path = "target_image.jpg"
    
    print("開始處理圖片...")
    query_patches, query_size = get_patch_embeddings(query_path)
    target_patches, _ = get_patch_embeddings(target_path)
    
    print("計算相似度熱圖...")
    heatmap = compute_similarity_heatmap(query_patches, target_patches)
    
    print("生成並儲存視覺化結果...")
    visualize_heatmap(query_path, heatmap)
    print("完成！請下載 heatmap_result.png 查看結果。")