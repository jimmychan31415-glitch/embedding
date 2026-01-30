import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

# Load image
image = Image.open("better_dog_cat.jpg").convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device).float()
print(f"Input shape: {input_tensor.shape}")  # Should be [1, 3, 336, 336]

# Direct forward through visual encoder
with torch.no_grad():
    # Step through visual encoder
    visual = model.visual
    
    # Initial projection
    x = visual.conv1(input_tensor)
    print(f"After conv1: {x.shape}")
    
    x = x.reshape(x.shape[0], x.shape[1], -1)
    print(f"After reshape to sequence: {x.shape}")  # [B, C, num_patches]
    
    x = x.permute(0, 2, 1)
    print(f"After permute to [B, L, C]: {x.shape}")  # [B, num_patches, C]
    
    # Add class token
    x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device),
                   x], dim=1)
    print(f"After adding class token: {x.shape}")
    
    # Add positional embeddings and pass through transformer
    x = x + visual.positional_embedding.to(x.dtype)
    print(f"After adding pos_embed: {x.shape}")
    
    # Check the transformer structure
    print(f"\nNumber of resblocks: {len(visual.transformer.resblocks)}")
    
    # Pass through first block
    x = visual.transformer.resblocks[0](x)
    print(f"After resblock[0]: {x.shape}")
    
    # Pass through last block
    x_final = visual.transformer.resblocks[-1](x if len(visual.transformer.resblocks) == 1 else 
                                                visual.transformer(visual.transformer.resblocks[0](x) 
                                                if len(visual.transformer.resblocks) > 1 else x))
    print(f"After all resblocks: {x.shape}")