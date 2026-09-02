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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertModel,
)
from PIL import Image
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH    = os.path.join(ROOT, "knowledge_base", "campus_knowledge.json")
CSV_PATH   = os.path.join(ROOT, "dataset", "text_dataset", "campus_queries.csv")
IMG_DIR    = os.path.join(ROOT, "dataset", "image_dataset")
AUD_DIR    = os.path.join(ROOT, "dataset", "audio_dataset")
CLIP_DIR   = os.path.join(ROOT, "saved_models", "clip")
FAISS_DIR  = os.path.join(ROOT, "saved_models", "faiss")
BERT_DIR   = os.path.join(ROOT, "saved_models", "distilbert")
FUSION_DIR = os.path.join(ROOT, "saved_models", "fusion")
GRAPH_DIR  = os.path.join(ROOT, "outputs", "graphs")
OUT_DIR    = os.path.join(ROOT, "outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {DEVICE}")

# Load knowledge base
with open(KB_PATH, encoding="utf-8") as f:
    kb = json.load(f)
class_names = [r["class"] for r in kb]
class_to_idx = {r["class"]: i for i, r in enumerate(kb)}
name_to_class = {r["name"]: r["class"] for r in kb}


# ═══════════════════════════════════════════════════════════════
# 1.  CLIP + FAISS EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  1.  CLIP + FAISS IMAGE RETRIEVAL EVALUATION")
print("="*70)

import faiss

# Load artefacts
with open(os.path.join(CLIP_DIR, "image_embeddings.pkl"), "rb") as f:
    clip_data = pickle.load(f)
image_embeddings = clip_data["embeddings"]
image_labels     = clip_data["labels"]

with open(os.path.join(CLIP_DIR, "text_embeddings.pkl"), "rb") as f:
    text_embeddings = pickle.load(f)

index = faiss.read_index(os.path.join(FAISS_DIR, "campus_index.faiss"))

# Evaluate top-1 and top-3
k = 5
distances, indices = index.search(image_embeddings, k)

top1_correct = (indices[:, 0] == image_labels).sum()
top3_correct = sum(1 for i in range(len(image_labels)) if image_labels[i] in indices[i, :3])
top5_correct = sum(1 for i in range(len(image_labels)) if image_labels[i] in indices[i, :5])

n = len(image_labels)
clip_top1 = top1_correct / n * 100
clip_top3 = top3_correct / n * 100
clip_top5 = top5_correct / n * 100

print(f"\n  Total images evaluated : {n}")
print(f"  Top-1 Accuracy        : {clip_top1:.2f}%")
print(f"  Top-3 Accuracy        : {clip_top3:.2f}%")
print(f"  Top-5 Accuracy        : {clip_top5:.2f}%")


# ═══════════════════════════════════════════════════════════════
# 2.  WHISPER WER EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  2.  WHISPER ASR — WER SUMMARY")
print("="*70)

trans_path = os.path.join(OUT_DIR, "whisper_transcriptions.csv")
if os.path.exists(trans_path):
    trans_df = pd.read_csv(trans_path)
    from jiwer import wer as compute_wer

    wer_scores = []
    for _, row in trans_df.iterrows():
        if pd.notna(row.get("ground_truth", "")) and str(row["ground_truth"]).strip():
            w = compute_wer(str(row["ground_truth"]).lower(),
                           str(row["whisper_transcript"]).lower())
            wer_scores.append(w)

    mean_wer = np.mean(wer_scores) * 100
    median_wer = np.median(wer_scores) * 100
    perfect = sum(1 for w in wer_scores if w == 0)

    print(f"\n  Files evaluated    : {len(wer_scores)}")
    print(f"  Mean WER           : {mean_wer:.2f}%")
    print(f"  Median WER         : {median_wer:.2f}%")
    print(f"  Perfect (0% WER)   : {perfect}/{len(wer_scores)}")
else:
    mean_wer = -1
    print("  [!] Whisper transcriptions not found — run 04_audio_processing_whisper.py first")


# ═══════════════════════════════════════════════════════════════
# 3.  DISTILBERT INTENT EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  3.  DISTILBERT INTENT CLASSIFICATION")
print("="*70)

df = pd.read_csv(CSV_PATH)
df["query_clean"] = df["query"].str.lower().str.strip()

# Load label maps
with open(os.path.join(BERT_DIR, "label_maps.json")) as f:
    label_maps = json.load(f)
intent2id = label_maps["intent2id"]
id2intent = {int(k): v for k, v in label_maps["id2intent"].items()}
loc2id    = label_maps["loc2id"]
id2loc    = {int(k): v for k, v in label_maps["id2loc"].items()}

# Load intent model
intent_model_path = os.path.join(BERT_DIR, "intent")
if os.path.exists(intent_model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(intent_model_path)
    intent_model = DistilBertForSequenceClassification.from_pretrained(intent_model_path).to(DEVICE)
    intent_model.eval()

    # Predict on full dataset
    all_intent_preds = []
    all_intent_true  = []

    for i in tqdm(range(0, len(df), 32), desc="  Intent prediction"):
        batch = df.iloc[i:i+32]
        tokens = tokenizer(
            batch["query_clean"].tolist(), padding=True,
            truncation=True, max_length=64, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = intent_model(**tokens).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
        all_intent_preds.extend(preds)
        all_intent_true.extend(batch["intent"].map(intent2id).tolist())

    intent_acc = accuracy_score(all_intent_true, all_intent_preds) * 100
    i_prec, i_rec, i_f1, _ = precision_recall_fscore_support(
        all_intent_true, all_intent_preds, average="macro"
    )

    target_names = [id2intent[i] for i in range(len(intent2id))]
    print(f"\n{classification_report(all_intent_true, all_intent_preds, target_names=target_names)}")
    print(f"  Overall Accuracy : {intent_acc:.2f}%")
    print(f"  Macro F1         : {i_f1*100:.2f}%")
else:
    intent_acc = -1
    i_f1 = -1
    print("  [!] Intent model not found — run 05_train_distilbert.py first")


# ═══════════════════════════════════════════════════════════════
# 4.  DISTILBERT LOCATION EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  4.  DISTILBERT LOCATION ENTITY CLASSIFICATION")
print("="*70)

location_model_path = os.path.join(BERT_DIR, "location")
if os.path.exists(location_model_path):
    loc_model = DistilBertForSequenceClassification.from_pretrained(location_model_path).to(DEVICE)
    loc_model.eval()

    all_loc_preds = []
    all_loc_true  = []

    for i in tqdm(range(0, len(df), 32), desc="  Location prediction"):
        batch = df.iloc[i:i+32]
        tokens = tokenizer(
            batch["query_clean"].tolist(), padding=True,
            truncation=True, max_length=64, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = loc_model(**tokens).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
        all_loc_preds.extend(preds)
        all_loc_true.extend(batch["location"].map(loc2id).tolist())

    loc_acc = accuracy_score(all_loc_true, all_loc_preds) * 100
    l_prec, l_rec, l_f1, _ = precision_recall_fscore_support(
        all_loc_true, all_loc_preds, average="macro"
    )
    print(f"\n  Location Accuracy : {loc_acc:.2f}%")
    print(f"  Location F1       : {l_f1*100:.2f}%")
else:
    loc_acc = -1
    l_f1 = -1
    print("  [!] Location model not found")


# ═══════════════════════════════════════════════════════════════
# 5.  FUSION MLP EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  5.  FUSION MLP — END-TO-END EVALUATION")
print("="*70)

# Import FusionMLP class
sys.path.insert(0, os.path.join(ROOT, "src"))
from importlib import import_module

# Re-define FusionMLP inline to avoid import issues
class FusionMLP(nn.Module):
    def __init__(self, clip_dim=512, bert_dim=768, num_classes=20):
        super().__init__()
        self.clip_dim  = clip_dim
        self.bert_dim  = bert_dim
        self.image_proj = nn.Sequential(nn.Linear(clip_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.text_proj  = nn.Sequential(nn.Linear(bert_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.image_gate = nn.Sequential(nn.Linear(clip_dim, 1), nn.Sigmoid())
        self.text_gate  = nn.Sequential(nn.Linear(bert_dim, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_emb, text_emb, image_mask=None, text_mask=None):
        img_gate = self.image_gate(image_emb)
        txt_gate = self.text_gate(text_emb)
        if image_mask is not None:
            img_gate = img_gate * image_mask
        if text_mask is not None:
            txt_gate = txt_gate * text_mask
        img_proj = self.image_proj(image_emb) * img_gate
        txt_proj = self.text_proj(text_emb)   * txt_gate
        fused = torch.cat([img_proj, txt_proj], dim=1)
        return self.classifier(fused)

fusion_path = os.path.join(FUSION_DIR, "fusion_model.pt")
if os.path.exists(fusion_path):
    fusion_model = FusionMLP().to(DEVICE)
    fusion_model.load_state_dict(torch.load(fusion_path, map_location=DEVICE, weights_only=True))
    fusion_model.eval()

    # Evaluate on image-only samples
    img_preds  = []
    img_trues  = []
    for i in range(len(image_embeddings)):
        img_e = torch.tensor(image_embeddings[i]).unsqueeze(0).to(DEVICE)
        txt_e = torch.zeros(1, 768).to(DEVICE)
        img_m = torch.ones(1, 1).to(DEVICE)
        txt_m = torch.zeros(1, 1).to(DEVICE)
        with torch.no_grad():
            logits = fusion_model(img_e, txt_e, img_m, txt_m)
            pred = logits.argmax(dim=1).item()
        img_preds.append(pred)
        img_trues.append(image_labels[i])

    fusion_img_acc = accuracy_score(img_trues, img_preds) * 100
    print(f"\n  Fusion (image-only)  : {fusion_img_acc:.2f}%")

    # Evaluate on text-only samples (use DistilBERT base for embeddings)
    bert_base = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    bert_base.eval()

    txt_preds  = []
    txt_trues  = []
    for i in tqdm(range(0, len(df), 32), desc="  Fusion text-only"):
        batch = df.iloc[i:i+32]
        tokens = tokenizer(
            batch["query_clean"].tolist(), padding=True,
            truncation=True, max_length=64, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            bert_out = bert_base(**tokens)
            cls_emb  = bert_out.last_hidden_state[:, 0, :]  # (B, 768)

            img_e = torch.zeros(cls_emb.size(0), 512).to(DEVICE)
            img_m = torch.zeros(cls_emb.size(0), 1).to(DEVICE)
            txt_m = torch.ones(cls_emb.size(0), 1).to(DEVICE)

            logits = fusion_model(img_e, cls_emb, img_m, txt_m)
            preds  = logits.argmax(dim=1).cpu().numpy()

        txt_preds.extend(preds)
        txt_trues.extend(batch["location"].map(class_to_idx).tolist())

    fusion_txt_acc = accuracy_score(txt_trues, txt_preds) * 100
    print(f"  Fusion (text-only)   : {fusion_txt_acc:.2f}%")
else:
    fusion_img_acc = -1
    fusion_txt_acc = -1
    print("  [!] Fusion model not found — run 06_train_fusion_model.py first")


# ═══════════════════════════════════════════════════════════════
# 6.  CONSOLIDATED RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  6.  CONSOLIDATED RESULTS")
print("="*70)

results = {
    "Component": [
        "CLIP + FAISS (Image Retrieval)",
        "CLIP + FAISS (Image Retrieval)",
        "CLIP + FAISS (Image Retrieval)",
        "Whisper ASR (Speech)",
        "DistilBERT (Intent Classification)",
        "DistilBERT (Intent Classification)",
        "DistilBERT (Location Entity)",
        "DistilBERT (Location Entity)",
        "Fusion MLP (Image-only)",
        "Fusion MLP (Text-only)",
    ],
    "Metric": [
        "Top-1 Accuracy",
        "Top-3 Accuracy",
        "Top-5 Accuracy",
        "Mean WER",
        "Accuracy",
        "Macro F1",
        "Accuracy",
        "Macro F1",
        "KB Retrieval Accuracy",
        "KB Retrieval Accuracy",
    ],
    "Value": [
        f"{clip_top1:.2f}%",
        f"{clip_top3:.2f}%",
        f"{clip_top5:.2f}%",
        f"{mean_wer:.2f}%" if mean_wer >= 0 else "N/A",
        f"{intent_acc:.2f}%" if intent_acc >= 0 else "N/A",
        f"{i_f1*100:.2f}%" if i_f1 >= 0 else "N/A",
        f"{loc_acc:.2f}%" if loc_acc >= 0 else "N/A",
        f"{l_f1*100:.2f}%" if l_f1 >= 0 else "N/A",
        f"{fusion_img_acc:.2f}%" if fusion_img_acc >= 0 else "N/A",
        f"{fusion_txt_acc:.2f}%" if fusion_txt_acc >= 0 else "N/A",
    ],
}

results_df = pd.DataFrame(results)
results_path = os.path.join(OUT_DIR, "results.csv")
results_df.to_csv(results_path, index=False)

print(f"\n{results_df.to_string(index=False)}")
print(f"\n  [✓] Results saved to: {results_path}")


# ═══════════════════════════════════════════════════════════════
# 7.  FINAL SUMMARY PLOT
# ═══════════════════════════════════════════════════════════════

# Bar chart of all component accuracies
fig, ax = plt.subplots(figsize=(12, 6))
components = ["CLIP\nTop-1", "CLIP\nTop-3", "Intent\nAcc", "Intent\nF1",
              "Location\nAcc", "Fusion\n(Image)", "Fusion\n(Text)"]
values = [
    clip_top1, clip_top3,
    intent_acc if intent_acc >= 0 else 0,
    i_f1*100 if i_f1 >= 0 else 0,
    loc_acc if loc_acc >= 0 else 0,
    fusion_img_acc if fusion_img_acc >= 0 else 0,
    fusion_txt_acc if fusion_txt_acc >= 0 else 0,
]
colors = ["#4472c4", "#5b9bd5", "#70ad47", "#a9d18e",
          "#ed7d31", "#9b59b6", "#e74c3c"]

bars = ax.bar(components, values, color=colors, edgecolor="white", width=0.6)
ax.set_ylabel("Accuracy / Score (%)")
ax.set_title("Multimodal Campus AI Assistant — Component Performance Summary")
ax.set_ylim(0, 105)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "system_performance_summary.png"), dpi=150)
plt.close()
print("  [✓] Saved: system_performance_summary.png")

print(f"""
{'='*70}
  EVALUATION COMPLETE
{'='*70}

  All results saved to : {results_path}
  All graphs saved to  : {GRAPH_DIR}

[✓] End-to-end system evaluation complete.
""")
