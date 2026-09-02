import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH     = os.path.join(ROOT, "knowledge_base", "campus_knowledge.json")
CSV_PATH    = os.path.join(ROOT, "dataset", "text_dataset", "campus_queries.csv")
IMG_DIR     = os.path.join(ROOT, "dataset", "image_dataset")
AUD_DIR     = os.path.join(ROOT, "dataset", "audio_dataset")
GRAPH_DIR   = os.path.join(ROOT, "outputs", "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)



# 1.  KNOWLEDGE BASE EXPLORATION

print("\n" + "="*70)
print("  1.  KNOWLEDGE BASE EXPLORATION")
print("="*70)

with open(KB_PATH, encoding="utf-8") as f:
    kb = json.load(f)

print(f"\n  Total records : {len(kb)}")
print(f"  Fields        : {list(kb[0].keys())}")
print(f"\n  ── Sample Record ──")
for k, v in kb[0].items():
    print(f"    {k:16s}: {v}")

# Category distribution
categories = [r["category"] for r in kb]
cat_counts = Counter(categories)
print(f"\n  Category distribution:")
for cat, cnt in cat_counts.most_common():
    print(f"    {cat:25s}  {cnt}")

fig, ax = plt.subplots(figsize=(10, 5))
cats = list(cat_counts.keys())
vals = list(cat_counts.values())
bars = ax.barh(cats, vals, color=sns.color_palette("viridis", len(cats)))
ax.set_xlabel("Number of Locations")
ax.set_title("Knowledge Base — Category Distribution")
ax.invert_yaxis()
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(v), va="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "kb_category_distribution.png"), dpi=150)
plt.close()
print("  [✓] Saved: kb_category_distribution.png")



# 2.  TEXT DATASET EXPLORATION

print("\n" + "="*70)
print("  2.  TEXT DATASET EXPLORATION")
print("="*70)

df = pd.read_csv(CSV_PATH)
print(f"\n  Total queries        : {len(df)}")
print(f"  Columns              : {list(df.columns)}")
print(f"  Unique intents       : {df['intent'].nunique()}  →  {sorted(df['intent'].unique())}")
print(f"  Unique locations     : {df['location'].nunique()}")

# ── 2a. Intent distribution ─────────────────────────────────
intent_counts = df["intent"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("Set2", len(intent_counts))
bars = ax.bar(intent_counts.index, intent_counts.values, color=colors)
ax.set_ylabel("Number of Queries")
ax.set_title("Text Dataset — Intent Distribution")
ax.set_xlabel("Intent Category")
for bar, v in zip(bars, intent_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "intent_distribution.png"), dpi=150)
plt.close()
print("  [✓] Saved: intent_distribution.png")

# ── 2b. Location distribution ───────────────────────────────
loc_counts = df["location"].value_counts()
fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("husl", len(loc_counts))
bars = ax.barh(loc_counts.index, loc_counts.values, color=colors)
ax.set_xlabel("Number of Queries")
ax.set_title("Text Dataset — Queries per Location")
ax.invert_yaxis()
for bar, v in zip(bars, loc_counts.values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            str(v), va="center", fontweight="bold", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "text_location_distribution.png"), dpi=150)
plt.close()
print("  [✓] Saved: text_location_distribution.png")

# ── 2c. Vocabulary statistics ────────────────────────────────
all_words = " ".join(df["query"].str.lower()).split()
vocab = set(all_words)
word_lengths = df["query"].apply(lambda x: len(x.split()))

print(f"\n  Total words          : {len(all_words)}")
print(f"  Unique vocabulary    : {len(vocab)}")
print(f"  Avg query length     : {word_lengths.mean():.1f} words")
print(f"  Min query length     : {word_lengths.min()} words")
print(f"  Max query length     : {word_lengths.max()} words")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(word_lengths, bins=range(word_lengths.min(), word_lengths.max()+2),
        color="#5b9bd5", edgecolor="white", alpha=0.85)
ax.set_xlabel("Query Length (words)")
ax.set_ylabel("Frequency")
ax.set_title("Text Dataset — Query Length Distribution")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "text_token_distribution.png"), dpi=150)
plt.close()
print("  [✓] Saved: text_token_distribution.png")

# ── 2d. Sample queries ──────────────────────────────────────
print("\n  ── Sample Queries by Intent ──")
for intent in sorted(df["intent"].unique()):
    samples = df[df["intent"] == intent]["query"].head(3).tolist()
    print(f"\n    [{intent}]")
    for s in samples:
        print(f"      • {s}")



# 3.  IMAGE DATASET EXPLORATION

print("\n" + "="*70)
print("  3.  IMAGE DATASET EXPLORATION")
print("="*70)

class_counts = {}
for cls in sorted(os.listdir(IMG_DIR)):
    cls_path = os.path.join(IMG_DIR, cls)
    if os.path.isdir(cls_path):
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        class_counts[cls] = len(imgs)

print(f"\n  Total classes  : {len(class_counts)}")
print(f"  Total images   : {sum(class_counts.values())}")
print(f"\n  Images per class:")
for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"    {cls:25s}  {cnt:4d}")

# ── 3a. Class distribution bar chart ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
classes = list(class_counts.keys())
counts  = list(class_counts.values())
colors  = sns.color_palette("coolwarm", len(classes))
bars    = ax.bar(classes, counts, color=colors, edgecolor="white")
ax.set_ylabel("Number of Images")
ax.set_title("Image Dataset — Class Distribution (Images per Location)")
ax.set_xlabel("Campus Location Class")
plt.xticks(rotation=45, ha="right")
for bar, v in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(v), ha="center", fontweight="bold", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "image_distribution.png"), dpi=150)
plt.close()
print("  [✓] Saved: image_distribution.png")

# ── 3b. Sample images grid (3 images × 6 classes) ───────────
sample_classes = list(class_counts.keys())[:6]
fig, axes = plt.subplots(len(sample_classes), 3, figsize=(12, 3*len(sample_classes)))
for row, cls in enumerate(sample_classes):
    cls_path = os.path.join(IMG_DIR, cls)
    imgs = sorted([f for f in os.listdir(cls_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:3]
    for col in range(3):
        ax = axes[row][col] if len(sample_classes) > 1 else axes[col]
        if col < len(imgs):
            img = Image.open(os.path.join(cls_path, imgs[col]))
            ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cls, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle("Image Dataset — Sample Images per Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "sample_images_grid.png"), dpi=150)
plt.close()
print("  [✓] Saved: sample_images_grid.png")



# 4.  AUDIO DATASET EXPLORATION

print("\n" + "="*70)
print("  4.  AUDIO DATASET EXPLORATION")
print("="*70)

try:
    import librosa

    audio_files = sorted([f for f in os.listdir(AUD_DIR) if f.endswith(".wav")])
    print(f"\n  Total audio files : {len(audio_files)}")

    durations = []
    sample_rates = []
    for af in audio_files:
        y, sr = librosa.load(os.path.join(AUD_DIR, af), sr=None)
        durations.append(librosa.get_duration(y=y, sr=sr))
        sample_rates.append(sr)

    durations = np.array(durations)
    print(f"  Total duration    : {durations.sum():.1f} seconds ({durations.sum()/60:.1f} min)")
    print(f"  Avg duration      : {durations.mean():.2f} s")
    print(f"  Min duration      : {durations.min():.2f} s")
    print(f"  Max duration      : {durations.max():.2f} s")
    print(f"  Sample rate       : {sample_rates[0]} Hz (consistent: {len(set(sample_rates)) == 1})")

    # Audio type breakdown
    types = Counter()
    for af in audio_files:
        for tag in ["direction", "hours", "event", "info"]:
            if tag in af:
                types[tag] += 1
                break
    print(f"\n  Audio query types:")
    for t, c in types.most_common():
        print(f"    {t:15s}  {c}")

    # ── 4a. Duration histogram ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(durations, bins=15, color="#70ad47", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Number of Files")
    ax.set_title("Audio Dataset — Duration Distribution")
    ax.axvline(durations.mean(), color="red", linestyle="--", label=f"Mean: {durations.mean():.2f}s")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "audio_duration_histogram.png"), dpi=150)
    plt.close()
    print("  [✓] Saved: audio_duration_histogram.png")

    # ── 4b. MFCC visualisation (3 samples) ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, af in enumerate(audio_files[:3]):
        y, sr = librosa.load(os.path.join(AUD_DIR, af), sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfcc, x_axis="time", ax=axes[i], sr=sr)
        axes[i].set_title(af.replace(".wav", ""), fontsize=9)
        axes[i].set_ylabel("MFCC Coeff" if i == 0 else "")
    fig.suptitle("Audio Dataset — MFCC Feature Visualisation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "mfcc_samples.png"), dpi=150)
    plt.close()
    print("  [✓] Saved: mfcc_samples.png")

except ImportError:
    print("  [!] librosa not installed — skipping audio exploration")
    print("      Install with:  pip install librosa soundfile")



# 5.  SUMMARY

print("\n" + "="*70)
print("  EXPLORATION SUMMARY")
print("="*70)
print(f"""
  Knowledge Base   :  {len(kb)} location records
  Text Queries     :  {len(df)} queries, {df['intent'].nunique()} intents, {df['location'].nunique()} locations
  Image Dataset    :  {sum(class_counts.values())} images across {len(class_counts)} classes
  Audio Dataset    :  {len(audio_files) if 'audio_files' in dir() else 'N/A'} .wav files

  All plots saved to:  {GRAPH_DIR}
""")
print("[✓] Data exploration complete.\n")
