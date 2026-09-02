import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH   = os.path.join(ROOT, "knowledge_base", "campus_knowledge.json")
IMG_DIR   = os.path.join(ROOT, "dataset", "image_dataset")
GRAPH_DIR = os.path.join(ROOT, "outputs", "graphs")
CLIP_DIR  = os.path.join(ROOT, "saved_models", "clip")
FAISS_DIR = os.path.join(ROOT, "saved_models", "faiss")

for d in [GRAPH_DIR, CLIP_DIR, FAISS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# 1.  LOAD CLIP MODEL
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  1.  LOADING CLIP MODEL (ViT-B/32)")
print("="*70)

try:
    import clip
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    print("  [✓] CLIP loaded successfully")
except ImportError:
    print("  [!] CLIP not installed. Install with:")
    print("      pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 2.  LOAD KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  2.  ENCODING KNOWLEDGE BASE TEXT DESCRIPTIONS")
print("="*70)

with open(KB_PATH, encoding="utf-8") as f:
    kb = json.load(f)

# Create rich text descriptions for each KB record
kb_texts = []
kb_classes = []
kb_names  = []
for record in kb:
    # Combine name + category + description for richer embedding
    text = f"{record['name']}: {record['description']} Category: {record['category']}"
    kb_texts.append(text)
    kb_classes.append(record["class"])
    kb_names.append(record["name"])

print(f"  KB records: {len(kb_texts)}")

# Encode text descriptions with CLIP
with torch.no_grad():
    text_tokens = clip.tokenize(kb_texts, truncate=True).to(DEVICE)
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings = text_embeddings.cpu().numpy().astype("float32")

# L2-normalise for cosine similarity via inner product
norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
text_embeddings_norm = text_embeddings / norms

print(f"  Text embedding shape: {text_embeddings_norm.shape}")

# Save text embeddings and mapping
with open(os.path.join(CLIP_DIR, "text_embeddings.pkl"), "wb") as f:
    pickle.dump(text_embeddings_norm, f)

kb_mapping = {i: {"class": kb_classes[i], "name": kb_names[i]} for i in range(len(kb))}
with open(os.path.join(CLIP_DIR, "kb_mapping.json"), "w") as f:
    json.dump(kb_mapping, f, indent=2)
print("  [✓] Text embeddings and mapping saved")


# ═══════════════════════════════════════════════════════════════
# 3.  BUILD FAISS INDEX
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  3.  BUILDING FAISS INDEX")
print("="*70)

import faiss

embedding_dim = text_embeddings_norm.shape[1]
index = faiss.IndexFlatIP(embedding_dim)   # inner product = cosine on normalised
index.add(text_embeddings_norm)

faiss_path = os.path.join(FAISS_DIR, "campus_index.faiss")
faiss.write_index(index, faiss_path)
print(f"  Index size      : {index.ntotal} vectors")
print(f"  Embedding dim   : {embedding_dim}")
print(f"  [✓] FAISS index saved to: {faiss_path}")


# ═══════════════════════════════════════════════════════════════
# 4.  ENCODE ALL IMAGES
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  4.  ENCODING ALL DATASET IMAGES")
print("="*70)

image_embeddings = []
image_labels     = []
image_paths      = []

# Build class-to-KB-index mapping
class_to_kb_idx = {record["class"]: i for i, record in enumerate(kb)}

class_dirs = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
total_images = 0

for cls in class_dirs:
    cls_path = os.path.join(IMG_DIR, cls)
    img_files = sorted([f for f in os.listdir(cls_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    total_images += len(img_files)

print(f"  Total images to encode: {total_images}")

for cls in tqdm(class_dirs, desc="  Encoding classes"):
    cls_path = os.path.join(IMG_DIR, cls)
    img_files = sorted([f for f in os.listdir(cls_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])

    kb_idx = class_to_kb_idx.get(cls, -1)

    for fname in img_files:
        img_path = os.path.join(cls_path, fname)
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb.cpu().numpy().astype("float32")
                emb = emb / np.linalg.norm(emb)
            image_embeddings.append(emb.flatten())
            image_labels.append(kb_idx)
            image_paths.append(img_path)
        except Exception as e:
            print(f"    [!] Skipped {fname}: {e}")

image_embeddings = np.array(image_embeddings, dtype="float32")
image_labels     = np.array(image_labels)

print(f"\n  Encoded images   : {len(image_embeddings)}")
print(f"  Embedding shape  : {image_embeddings.shape}")

# Save image embeddings
with open(os.path.join(CLIP_DIR, "image_embeddings.pkl"), "wb") as f:
    pickle.dump({
        "embeddings": image_embeddings,
        "labels": image_labels,
        "paths": image_paths,
        "class_to_kb_idx": class_to_kb_idx
    }, f)
print("  [✓] Image embeddings saved")


# ═══════════════════════════════════════════════════════════════
# 5.  EVALUATE RETRIEVAL ACCURACY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  5.  RETRIEVAL EVALUATION")
print("="*70)

# For each image embedding, query FAISS for top-3 nearest text embeddings
k = 3
distances, indices = index.search(image_embeddings, k)

top1_correct = 0
top3_correct = 0
total = len(image_embeddings)

per_class_correct = {cls: {"top1": 0, "top3": 0, "total": 0} for cls in class_dirs}

for i in range(total):
    true_label = image_labels[i]
    predicted_top1 = indices[i][0]
    predicted_top3 = indices[i][:3]

    # Find class name from path
    cls_name = os.path.basename(os.path.dirname(image_paths[i]))

    per_class_correct[cls_name]["total"] += 1

    if predicted_top1 == true_label:
        top1_correct += 1
        per_class_correct[cls_name]["top1"] += 1

    if true_label in predicted_top3:
        top3_correct += 1
        per_class_correct[cls_name]["top3"] += 1

top1_acc = top1_correct / total * 100
top3_acc = top3_correct / total * 100

print(f"\n  ── Overall Retrieval Accuracy ──")
print(f"    Top-1 Accuracy : {top1_acc:.2f}%  ({top1_correct}/{total})")
print(f"    Top-3 Accuracy : {top3_acc:.2f}%  ({top3_correct}/{total})")

print(f"\n  ── Per-Class Accuracy ──")
print(f"    {'Class':25s}  {'Top-1':>8s}  {'Top-3':>8s}  {'Total':>6s}")
print(f"    {'─'*25}  {'─'*8}  {'─'*8}  {'─'*6}")
for cls in class_dirs:
    d = per_class_correct[cls]
    t1 = d["top1"]/d["total"]*100 if d["total"] > 0 else 0
    t3 = d["top3"]/d["total"]*100 if d["total"] > 0 else 0
    print(f"    {cls:25s}  {t1:7.1f}%  {t3:7.1f}%  {d['total']:>6d}")


# ═══════════════════════════════════════════════════════════════
# 6.  SAVE RESULTS & PLOTS
# ═══════════════════════════════════════════════════════════════

# ── 6a. Per-class accuracy bar chart ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
classes_sorted = sorted(per_class_correct.keys())
t1_vals = [per_class_correct[c]["top1"]/per_class_correct[c]["total"]*100
           if per_class_correct[c]["total"] > 0 else 0 for c in classes_sorted]
t3_vals = [per_class_correct[c]["top3"]/per_class_correct[c]["total"]*100
           if per_class_correct[c]["total"] > 0 else 0 for c in classes_sorted]

x = np.arange(len(classes_sorted))
width = 0.35
bars1 = ax.bar(x - width/2, t1_vals, width, label="Top-1", color="#4472c4")
bars2 = ax.bar(x + width/2, t3_vals, width, label="Top-3", color="#70ad47")
ax.set_ylabel("Accuracy (%)")
ax.set_title("CLIP + FAISS — Retrieval Accuracy per Class")
ax.set_xticks(x)
ax.set_xticklabels(classes_sorted, rotation=45, ha="right")
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "clip_retrieval_accuracy.png"), dpi=150)
plt.close()
print("\n  [✓] Saved: clip_retrieval_accuracy.png")

# ── 6b. Results CSV ──────────────────────────────────────────
results_path = os.path.join(ROOT, "outputs", "results.csv")
results_data = {
    "Component": ["CLIP + FAISS (Vision)"],
    "Metric": ["Retrieval Accuracy"],
    "Top-1 (%)": [round(top1_acc, 2)],
    "Top-3 (%)": [round(top3_acc, 2)],
}
results_df = pd.DataFrame(results_data)
results_df.to_csv(results_path, index=False)
print(f"  [✓] Results saved to: {results_path}")

print(f"""
  ── Summary ──
  CLIP Model     : ViT-B/32 (frozen, zero-shot)
  FAISS Index    : {index.ntotal} text vectors, dim={embedding_dim}
  Images Encoded : {len(image_embeddings)}
  Top-1 Accuracy : {top1_acc:.2f}%
  Top-3 Accuracy : {top3_acc:.2f}%

[✓] CLIP + FAISS pipeline complete.
""")
