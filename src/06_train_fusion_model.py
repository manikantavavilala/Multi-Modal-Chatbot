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
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH    = os.path.join(ROOT, "knowledge_base", "campus_knowledge.json")
CSV_PATH   = os.path.join(ROOT, "dataset", "text_dataset", "campus_queries.csv")
CLIP_DIR   = os.path.join(ROOT, "saved_models", "clip")
BERT_DIR   = os.path.join(ROOT, "saved_models", "distilbert")
FUSION_DIR = os.path.join(ROOT, "saved_models", "fusion")
GRAPH_DIR  = os.path.join(ROOT, "outputs", "graphs")
OUT_DIR    = os.path.join(ROOT, "outputs")

for d in [FUSION_DIR, GRAPH_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {DEVICE}")

# Dimensions
CLIP_DIM = 512
BERT_DIM = 768
NUM_CLASSES = 20


# ═══════════════════════════════════════════════════════════════
# 1.  FUSION MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class FusionMLP(nn.Module):
    """
    Multimodal fusion model that combines image (CLIP) and text (DistilBERT)
    embeddings to predict the correct knowledge-base location.

    Routing strategy:
      - Each modality input is independently sufficient
      - Absent modalities are masked with learned zero vectors
      - An attention-like gating mechanism weights available modalities
    """
    def __init__(self, clip_dim=512, bert_dim=768, num_classes=20):
        super().__init__()
        self.clip_dim  = clip_dim
        self.bert_dim  = bert_dim
        combined_dim   = clip_dim + bert_dim  # 1280

        # Modality-specific projection layers
        self.image_proj = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Gating mechanism — learns to weight modalities
        self.image_gate = nn.Sequential(nn.Linear(clip_dim, 1), nn.Sigmoid())
        self.text_gate  = nn.Sequential(nn.Linear(bert_dim, 1), nn.Sigmoid())

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_emb, text_emb, image_mask=None, text_mask=None):
        """
        Args:
            image_emb:  (B, 512) CLIP image embedding
            text_emb:   (B, 768) DistilBERT [CLS] embedding
            image_mask: (B, 1) — 1.0 if image present, 0.0 if absent
            text_mask:  (B, 1) — 1.0 if text present, 0.0 if absent
        """
        # Apply gating
        img_gate = self.image_gate(image_emb)  # (B, 1)
        txt_gate = self.text_gate(text_emb)     # (B, 1)

        # Apply modality masks (zero out absent modalities)
        if image_mask is not None:
            img_gate = img_gate * image_mask
        if text_mask is not None:
            txt_gate = txt_gate * text_mask

        # Project each modality
        img_proj = self.image_proj(image_emb) * img_gate  # (B, 256)
        txt_proj = self.text_proj(text_emb)   * txt_gate  # (B, 256)

        # Concatenate projected features
        fused = torch.cat([img_proj, txt_proj], dim=1)     # (B, 512)

        # Classify
        logits = self.classifier(fused)                    # (B, 20)
        return logits


# ═══════════════════════════════════════════════════════════════
# 2.  GENERATE TRAINING DATA
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  2.  GENERATING FUSION TRAINING DATA")
print("="*70)

# Load knowledge base
with open(KB_PATH, encoding="utf-8") as f:
    kb = json.load(f)
class_to_kb_idx = {r["class"]: i for i, r in enumerate(kb)}

# ── 2a. Load image embeddings from CLIP ──────────────────────
with open(os.path.join(CLIP_DIR, "image_embeddings.pkl"), "rb") as f:
    clip_data = pickle.load(f)
image_embeddings = clip_data["embeddings"]     # (N_img, 512)
image_labels     = clip_data["labels"]          # (N_img,)

print(f"  Image embeddings : {image_embeddings.shape}")

# ── 2b. Generate text embeddings from DistilBERT ────────────
print("  Generating DistilBERT [CLS] embeddings for text queries...")
df = pd.read_csv(CSV_PATH)
df["query_clean"] = df["query"].str.lower().str.strip()

# Load DistilBERT for feature extraction (not the classifier head)
tokenizer_bert = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
bert_model.eval()

# Label mappings
loc2id = {r["class"]: i for i, r in enumerate(kb)}

text_embeddings_list = []
text_labels_list     = []

batch_size = 32
queries = df["query_clean"].tolist()
locations = df["location"].tolist()

for i in tqdm(range(0, len(queries), batch_size), desc="  Encoding text"):
    batch_queries = queries[i:i+batch_size]
    batch_locs    = locations[i:i+batch_size]

    tokens = tokenizer_bert(
        batch_queries, padding=True, truncation=True,
        max_length=64, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token

    text_embeddings_list.append(cls_emb.cpu().numpy())
    text_labels_list.extend([loc2id.get(loc, -1) for loc in batch_locs])

text_embeddings = np.concatenate(text_embeddings_list, axis=0).astype(np.float32)
text_labels     = np.array(text_labels_list)

print(f"  Text embeddings  : {text_embeddings.shape}")

# ── 2c. Build training samples ──────────────────────────────
print("\n  Building training samples (image-only, text-only, combined)...")

all_image_embs   = []
all_text_embs    = []
all_image_masks  = []
all_text_masks   = []
all_labels       = []

# Group embeddings by class
img_by_class  = {}
text_by_class = {}

for i in range(len(image_embeddings)):
    lbl = image_labels[i]
    if lbl not in img_by_class:
        img_by_class[lbl] = []
    img_by_class[lbl].append(image_embeddings[i])

for i in range(len(text_embeddings)):
    lbl = text_labels[i]
    if lbl not in text_by_class:
        text_by_class[lbl] = []
    text_by_class[lbl].append(text_embeddings[i])

# --- Image-only samples ---
for i in range(len(image_embeddings)):
    all_image_embs.append(image_embeddings[i])
    all_text_embs.append(np.zeros(BERT_DIM, dtype=np.float32))
    all_image_masks.append(1.0)
    all_text_masks.append(0.0)
    all_labels.append(image_labels[i])

# --- Text-only samples ---
for i in range(len(text_embeddings)):
    all_image_embs.append(np.zeros(CLIP_DIM, dtype=np.float32))
    all_text_embs.append(text_embeddings[i])
    all_image_masks.append(0.0)
    all_text_masks.append(1.0)
    all_labels.append(text_labels[i])

# --- Combined samples (pair random image + text of same class) ---
np.random.seed(42)
for cls_idx in range(NUM_CLASSES):
    cls_imgs  = img_by_class.get(cls_idx, [])
    cls_texts = text_by_class.get(cls_idx, [])
    if not cls_imgs or not cls_texts:
        continue
    n_combined = min(len(cls_imgs), len(cls_texts), 50)
    for j in range(n_combined):
        img_idx  = np.random.randint(0, len(cls_imgs))
        text_idx = np.random.randint(0, len(cls_texts))
        all_image_embs.append(cls_imgs[img_idx])
        all_text_embs.append(cls_texts[text_idx])
        all_image_masks.append(1.0)
        all_text_masks.append(1.0)
        all_labels.append(cls_idx)

# Convert to tensors
all_image_embs  = torch.tensor(np.array(all_image_embs), dtype=torch.float32)
all_text_embs   = torch.tensor(np.array(all_text_embs), dtype=torch.float32)
all_image_masks = torch.tensor(np.array(all_image_masks), dtype=torch.float32).unsqueeze(1)
all_text_masks  = torch.tensor(np.array(all_text_masks), dtype=torch.float32).unsqueeze(1)
all_labels      = torch.tensor(np.array(all_labels), dtype=torch.long)

print(f"  Total samples    : {len(all_labels)}")
print(f"    Image-only     : {len(image_embeddings)}")
print(f"    Text-only      : {len(text_embeddings)}")
print(f"    Combined       : {len(all_labels) - len(image_embeddings) - len(text_embeddings)}")

# ── 2d. Train/val split ──────────────────────────────────────
from sklearn.model_selection import train_test_split

indices = list(range(len(all_labels)))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=all_labels.numpy()
)

train_dataset = TensorDataset(
    all_image_embs[train_idx], all_text_embs[train_idx],
    all_image_masks[train_idx], all_text_masks[train_idx],
    all_labels[train_idx]
)
val_dataset = TensorDataset(
    all_image_embs[val_idx], all_text_embs[val_idx],
    all_image_masks[val_idx], all_text_masks[val_idx],
    all_labels[val_idx]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"  Train set        : {len(train_dataset)}")
print(f"  Val set          : {len(val_dataset)}")


# ═══════════════════════════════════════════════════════════════
# 3.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  3.  TRAINING FUSION MLP")
print("="*70)

model = FusionMLP(clip_dim=CLIP_DIM, bert_dim=BERT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

NUM_EPOCHS    = 30
PATIENCE      = 5
best_val_acc  = 0.0
patience_cnt  = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total   = 0

    for img_e, txt_e, img_m, txt_m, labels in train_loader:
        img_e  = img_e.to(DEVICE)
        txt_e  = txt_e.to(DEVICE)
        img_m  = img_m.to(DEVICE)
        txt_m  = txt_m.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(img_e, txt_e, img_m, txt_m)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    scheduler.step()

    # ── Validate ─────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for img_e, txt_e, img_m, txt_m, labels in val_loader:
            img_e  = img_e.to(DEVICE)
            txt_e  = txt_e.to(DEVICE)
            img_m  = img_m.to(DEVICE)
            txt_m  = txt_m.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(img_e, txt_e, img_m, txt_m)
            loss   = criterion(logits, labels)

            val_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    epoch_train_loss = train_loss / train_total
    epoch_val_loss   = val_loss / val_total
    epoch_train_acc  = train_correct / train_total * 100
    epoch_val_acc    = val_correct / val_total * 100

    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_acc"].append(epoch_val_acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"Loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f}  "
              f"Acc: {epoch_train_acc:.1f}%/{epoch_val_acc:.1f}%")

    # Early stopping
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_cnt = 0
        torch.save(model.state_dict(), os.path.join(FUSION_DIR, "fusion_model.pt"))
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best val acc: {best_val_acc:.1f}%)")
            break

print(f"\n  Best val accuracy: {best_val_acc:.1f}%")
print(f"  [✓] Model saved to: {os.path.join(FUSION_DIR, 'fusion_model.pt')}")

# Save model config
config = {
    "clip_dim": CLIP_DIM, "bert_dim": BERT_DIM,
    "num_classes": NUM_CLASSES,
    "architecture": "FusionMLP with gating",
}
with open(os.path.join(FUSION_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# 4.  PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  4.  GENERATING PLOTS")
print("="*70)

epochs_range = range(1, len(history["train_loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, history["train_loss"], "o-", color="#4472c4", label="Train Loss", markersize=3)
ax1.plot(epochs_range, history["val_loss"],   "s-", color="#ed7d31", label="Val Loss", markersize=3)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Fusion MLP — Loss Curve")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history["train_acc"], "o-", color="#70ad47", label="Train Acc", markersize=3)
ax2.plot(epochs_range, history["val_acc"],   "s-", color="#5b9bd5", label="Val Acc", markersize=3)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Fusion MLP — Accuracy Curve")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "fusion_loss_curve.png"), dpi=150)
plt.close()
print("  [✓] Saved: fusion_loss_curve.png")

# ── Confusion matrix on val set ──────────────────────────────
model.load_state_dict(torch.load(os.path.join(FUSION_DIR, "fusion_model.pt"),
                                  map_location=DEVICE, weights_only=True))
model.eval()

all_preds  = []
all_trues  = []
with torch.no_grad():
    for img_e, txt_e, img_m, txt_m, labels in val_loader:
        logits = model(img_e.to(DEVICE), txt_e.to(DEVICE),
                       img_m.to(DEVICE), txt_m.to(DEVICE))
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_trues.extend(labels.numpy())

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)
fusion_acc = accuracy_score(all_trues, all_preds) * 100

# Class names
class_names = [r["class"] for r in kb]
cm = confusion_matrix(all_trues, all_preds)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Fusion MLP — Confusion Matrix (Val Acc: {fusion_acc:.1f}%)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "fusion_confusion_matrix.png"), dpi=150)
plt.close()
print("  [✓] Saved: fusion_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════
# 5.  UPDATE RESULTS
# ═══════════════════════════════════════════════════════════════
results_path = os.path.join(OUT_DIR, "results.csv")
try:
    results_df = pd.read_csv(results_path)
except FileNotFoundError:
    results_df = pd.DataFrame()

new_row = pd.DataFrame({
    "Component": ["Fusion MLP"],
    "Metric": [f"End-to-end KB Retrieval Accuracy"],
    "Top-1 (%)": [round(fusion_acc, 2)],
    "Top-3 (%)": ["N/A"],
})
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv(results_path, index=False)

print(f"""
  ── Summary ──
  Architecture    : FusionMLP (gated, 1280→512→256→128→20)
  Training data   : image-only + text-only + combined samples
  Best val acc    : {best_val_acc:.1f}%
  Final val acc   : {fusion_acc:.1f}%

[✓] Fusion model training complete.
""")
