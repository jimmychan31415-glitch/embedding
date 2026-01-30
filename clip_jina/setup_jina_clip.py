# setup_jina_clip.py - 設定 Jina-CLIP v2 模型（最後層輸出）

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device} (RTX 5090 D 已啟用)")

# 載入 Jina-CLIP v2（需 trust_remote_code=True）
jina_processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", use_fast=True, trust_remote_code=True)
jina_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device).eval()

def extract_embedding_jina(image):
    """提取 Jina-CLIP 最後層嵌入（不支援中間層）"""
    inputs = jina_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        return jina_model.get_image_features(**inputs)

# 測試
if __name__ == "__main__":
    image = Image.open("query_image.jpg").convert("RGB")
    
    emb_jina = extract_embedding_jina(image)
    
    print("Jina-CLIP 嵌入提取完成！")
    print(f"嵌入形狀：{emb_jina.shape}")