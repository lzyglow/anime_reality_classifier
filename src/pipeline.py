import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

def run_inference(image_dir, model_path="best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #Load model
    ckpt = torch.load(model_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Class mapping: {idx_to_class}")

    #rebuild model
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 2)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print("Model loaded.")

    #transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #get all image paths
    ROOT = image_dir  # zip1 folder
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    print(f"Scanning {ROOT} for images...")
    paths = []
    for root, _, files in os.walk(ROOT):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))

    print(f"Total images found: {len(paths)}")
    batch_size = 32
    rows = []

    for i in tqdm(range(0, len(paths), batch_size), desc="Processing"):
        batch_paths = paths[i:i+batch_size]
        batch_images = []
        batch_rel_ids = []

        #load batch size = 32
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(transform(img))
                batch_rel_ids.append(os.path.relpath(img_path, ROOT))
            except:
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            predictions = probs.argmax(dim=1).cpu().numpy()

        for rel_id, yi in zip(batch_rel_ids, predictions):
            rows.append({
                "image_id": rel_id,
                "label": idx_to_class[int(yi)]
            })
        print(i)

    #handle remaining images
    processed_ids = {row["image_id"] for row in rows}
    for img_path in paths:
        rel_id = os.path.relpath(img_path, ROOT)
        if rel_id not in processed_ids:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    yi = torch.softmax(outputs, dim=1).argmax().item()
                rows.append({
                    "image_id": rel_id,
                    "label": idx_to_class[int(yi)]
                })
            except:
                rows.append({
                    "image_id": rel_id,
                    "label": "error"
                })

    pred_df = pd.DataFrame(rows)
    #pred_df.to_parquet("./zip_predictions.parquet", index=False)

    return pred_df
