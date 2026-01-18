# evaluation_pipeline.py - 已修正類別不足問題

import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

print("載入 DINOv2...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = "./imagenette"

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

def extract_embeddings(loader, max_samples=500):
    embeddings = []
    labels = []
    count = 0
    
    for imgs, lbls in tqdm(loader, desc="提取嵌入"):
        imgs = imgs.to(device)
        with torch.no_grad():
            emb = model(pixel_values=imgs).last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu())
        labels.extend(lbls.tolist())
        
        count += imgs.shape[0]
        if count >= max_samples:
            excess = count - max_samples
            if excess > 0:
                embeddings[-1] = embeddings[-1][:-excess]
                labels = labels[:-excess]
            break
    
    embeddings = torch.cat(embeddings)
    labels = np.array(labels)
    print(f"提取完成：{len(embeddings)} 張圖片，類別數：{len(np.unique(labels))}")
    return embeddings, labels

print("提取訓練集...")
train_embeddings_dino, train_labels = extract_embeddings(train_loader, max_samples=1000)  # 增加訓練樣本

print("提取驗證集...")
val_embeddings_dino, val_labels = extract_embeddings(val_loader, max_samples=500)  # 增加驗證樣本

def compute_metrics(train_embeddings, train_labels, val_embeddings, val_labels):
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(train_embeddings.numpy(), train_labels)
    knn_acc = knn.score(val_embeddings.numpy(), val_labels)
    
    if len(np.unique(val_labels)) < 2:
        print("警告：驗證集只有 1 個類別，Silhouette Score 無法計算！")
        sil_score = np.nan
    else:
        sil_score = silhouette_score(val_embeddings.numpy(), val_labels)
    
    print(f"\n=== DINOv2 評估結果 ===")
    print(f"k-NN Accuracy (20): {knn_acc:.4f}")
    if not np.isnan(sil_score):
        print(f"Silhouette Score: {sil_score:.4f}")
    else:
        print("Silhouette Score: N/A (類別不足)")

print("\n開始計算指標...")
compute_metrics(train_embeddings_dino, train_labels, val_embeddings_dino, val_labels)

print("\n完成！")