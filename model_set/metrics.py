# metrics.py - 指標計算模組（可被其他檔案 import）

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

def compute_metrics(train_embeddings, train_labels, val_embeddings, val_labels):
    """
    計算基本 + 進階指標
    - train_embeddings / val_embeddings: torch.Tensor [N, dim]
    - train_labels / val_labels: np.array 或 list
    """
    print("計算 k-NN 準確率...")
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(train_embeddings.cpu().numpy(), train_labels)
    knn_acc = knn.score(val_embeddings.cpu().numpy(), val_labels)
    
    print("計算 Silhouette Score...")
    sil_score = silhouette_score(val_embeddings.cpu().numpy(), val_labels)
    
    print(f"\n=== 評估結果 ===")
    print(f"k-NN Accuracy (20): {knn_acc:.4f}")
    print(f"Silhouette Score: {sil_score:.4f}")
    
    return knn_acc, sil_score