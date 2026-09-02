import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(ROOT, "dataset", "text_dataset", "campus_queries.csv")
GRAPH_DIR  = os.path.join(ROOT, "outputs", "graphs")
MODEL_DIR  = os.path.join(ROOT, "saved_models", "distilbert")
OUT_DIR    = os.path.join(ROOT, "outputs")

for d in [GRAPH_DIR, MODEL_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {DEVICE}")



# 1.  LOAD & PREPARE DATA

print("\n" + "="*70)
print("  1.  LOADING & PREPARING DATA")
print("="*70)

df = pd.read_csv(CSV_PATH)
print(f"\n  Total samples : {len(df)}")
print(f"  Columns       : {list(df.columns)}")

# --- Text preprocessing ---
df["query_clean"] = df["query"].str.lower().str.strip()

# --- Encode intent labels ---
intent_labels = sorted(df["intent"].unique())
intent2id = {label: i for i, label in enumerate(intent_labels)}
id2intent = {i: label for label, i in intent2id.items()}
df["intent_id"] = df["intent"].map(intent2id)

# --- Encode location labels ---
location_labels = sorted(df["location"].unique())
loc2id = {label: i for i, label in enumerate(location_labels)}
id2loc = {i: label for label, i in loc2id.items()}
df["location_id"] = df["location"].map(loc2id)

print(f"  Intent classes  : {len(intent_labels)}  →  {intent_labels}")
print(f"  Location classes: {len(location_labels)}")

# --- Stratified train/val split ---
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["intent_id"]
)
print(f"\n  Train samples : {len(train_df)}")
print(f"  Val samples   : {len(val_df)}")

# Save label mappings
label_maps = {
    "intent2id": intent2id,
    "id2intent": {str(k): v for k, v in id2intent.items()},
    "loc2id": loc2id,
    "id2loc": {str(k): v for k, v in id2loc.items()},
}
with open(os.path.join(MODEL_DIR, "label_maps.json"), "w") as f:
    json.dump(label_maps, f, indent=2)
print("  [✓] Label mappings saved")



# 2.  TOKENISATION & DATASET

print("\n" + "="*70)
print("  2.  TOKENISATION")
print("="*70)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class CampusQueryDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=64, task="intent"):
        self.texts   = dataframe["query_clean"].tolist()
        if task == "intent":
            self.labels = dataframe["intent_id"].tolist()
        else:
            self.labels = dataframe["location_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Sample tokenisation output
sample = tokenizer(df["query_clean"].iloc[0], return_tensors="pt", max_length=64,
                   padding="max_length", truncation=True)
print(f"\n  Sample query   : \"{df['query_clean'].iloc[0]}\"")
print(f"  Token IDs      : {sample['input_ids'][0][:15].tolist()}...")
print(f"  Tokens         : {tokenizer.convert_ids_to_tokens(sample['input_ids'][0][:15])}")
print(f"  Max length     : 64")



# 3.  TRAIN INTENT CLASSIFIER

print("\n" + "="*70)
print("  3.  TRAINING INTENT CLASSIFIER")
print("="*70)

train_dataset_intent = CampusQueryDataset(train_df, tokenizer, task="intent")
val_dataset_intent   = CampusQueryDataset(val_df, tokenizer, task="intent")

model_intent = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(intent_labels),
)
model_intent.config.id2label = id2intent
model_intent.config.label2id = intent2id

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

training_args_intent = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "intent_checkpoints"),
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",      # disable WandB/TensorBoard
    fp16=torch.cuda.is_available(),
)

trainer_intent = Trainer(
    model=model_intent,
    args=training_args_intent,
    train_dataset=train_dataset_intent,
    eval_dataset=val_dataset_intent,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("  Starting intent training...")
train_result_intent = trainer_intent.train()
print("  [✓] Intent training complete")

# ── 3a. Extract & plot training history ──────────────────────
log_history = trainer_intent.state.log_history

# Separate train and eval logs
train_losses = []
eval_metrics  = {"epoch": [], "loss": [], "accuracy": [], "f1": []}

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_losses.append({"epoch": entry.get("epoch", 0), "loss": entry["loss"]})
    if "eval_loss" in entry:
        eval_metrics["epoch"].append(entry.get("epoch", 0))
        eval_metrics["loss"].append(entry["eval_loss"])
        eval_metrics["accuracy"].append(entry.get("eval_accuracy", 0))
        eval_metrics["f1"].append(entry.get("eval_f1", 0))

# Plot loss curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

if train_losses:
    ax1.plot([t["epoch"] for t in train_losses], [t["loss"] for t in train_losses],
             "o-", color="#4472c4", label="Train Loss")
if eval_metrics["loss"]:
    ax1.plot(eval_metrics["epoch"], eval_metrics["loss"],
             "s-", color="#ed7d31", label="Val Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("DistilBERT Intent — Loss Curve")
ax1.legend()
ax1.grid(True, alpha=0.3)

if eval_metrics["accuracy"]:
    ax2.plot(eval_metrics["epoch"], [a*100 for a in eval_metrics["accuracy"]],
             "o-", color="#70ad47", label="Accuracy")
    ax2.plot(eval_metrics["epoch"], [f*100 for f in eval_metrics["f1"]],
             "s-", color="#5b9bd5", label="F1 Score")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score (%)")
ax2.set_title("DistilBERT Intent — Accuracy & F1")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "distilbert_loss_curve.png"), dpi=150)
plt.close()
print("  [✓] Saved: distilbert_loss_curve.png")



# 4.  EVALUATE INTENT CLASSIFIER

print("\n" + "="*70)
print("  4.  INTENT CLASSIFIER EVALUATION")
print("="*70)

preds_output = trainer_intent.predict(val_dataset_intent)
preds = np.argmax(preds_output.predictions, axis=-1)
true_labels = np.array([val_dataset_intent[i]["labels"].item() for i in range(len(val_dataset_intent))])

# Classification report
target_names = [id2intent[i] for i in range(len(intent_labels))]
report = classification_report(true_labels, preds, target_names=target_names)
print(f"\n  Classification Report:\n{report}")

# Confusion matrix
cm = confusion_matrix(true_labels, preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("DistilBERT — Intent Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "intent_confusion_matrix.png"), dpi=150)
plt.close()
print("  [✓] Saved: intent_confusion_matrix.png")

# Overall metrics
intent_acc = accuracy_score(true_labels, preds)
intent_prec, intent_rec, intent_f1, _ = precision_recall_fscore_support(
    true_labels, preds, average="macro"
)
print(f"\n  Intent Accuracy  : {intent_acc*100:.2f}%")
print(f"  Intent Precision : {intent_prec*100:.2f}%")
print(f"  Intent Recall    : {intent_rec*100:.2f}%")
print(f"  Intent F1        : {intent_f1*100:.2f}%")



# 5.  TRAIN LOCATION ENTITY CLASSIFIER

print("\n" + "="*70)
print("  5.  TRAINING LOCATION ENTITY CLASSIFIER")
print("="*70)

train_dataset_loc = CampusQueryDataset(train_df, tokenizer, task="location")
val_dataset_loc   = CampusQueryDataset(val_df, tokenizer, task="location")

model_location = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(location_labels),
)

training_args_loc = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "location_checkpoints"),
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer_loc = Trainer(
    model=model_location,
    args=training_args_loc,
    train_dataset=train_dataset_loc,
    eval_dataset=val_dataset_loc,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("  Starting location training...")
train_result_loc = trainer_loc.train()
print("  [✓] Location training complete")

# Evaluate location classifier
preds_loc_output = trainer_loc.predict(val_dataset_loc)
preds_loc = np.argmax(preds_loc_output.predictions, axis=-1)
true_loc  = np.array([val_dataset_loc[i]["labels"].item() for i in range(len(val_dataset_loc))])

loc_acc = accuracy_score(true_loc, preds_loc)
loc_prec, loc_rec, loc_f1, _ = precision_recall_fscore_support(
    true_loc, preds_loc, average="macro"
)
print(f"\n  Location Accuracy  : {loc_acc*100:.2f}%")
print(f"  Location F1        : {loc_f1*100:.2f}%")

# Location confusion matrix
fig, ax = plt.subplots(figsize=(14, 11))
cm_loc = confusion_matrix(true_loc, preds_loc)
loc_names = [id2loc[i] for i in range(len(location_labels))]
sns.heatmap(cm_loc, annot=True, fmt="d", cmap="Greens",
            xticklabels=loc_names, yticklabels=loc_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("DistilBERT — Location Entity Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "location_confusion_matrix.png"), dpi=150)
plt.close()
print("  [✓] Saved: location_confusion_matrix.png")



# 6.  SAVE MODELS

print("\n" + "="*70)
print("  6.  SAVING MODELS")
print("="*70)

# Save intent model
intent_save_path = os.path.join(MODEL_DIR, "intent")
model_intent.save_pretrained(intent_save_path)
tokenizer.save_pretrained(intent_save_path)
print(f"  [✓] Intent model saved to: {intent_save_path}")

# Save location model
location_save_path = os.path.join(MODEL_DIR, "location")
model_location.save_pretrained(location_save_path)
tokenizer.save_pretrained(location_save_path)
print(f"  [✓] Location model saved to: {location_save_path}")



# 7.  UPDATE RESULTS

results_path = os.path.join(OUT_DIR, "results.csv")
try:
    results_df = pd.read_csv(results_path)
except FileNotFoundError:
    results_df = pd.DataFrame()

new_rows = pd.DataFrame({
    "Component": [
        "DistilBERT (Intent)",
        "DistilBERT (Location)",
    ],
    "Metric": [
        f"Acc={intent_acc*100:.1f}%, P={intent_prec*100:.1f}%, R={intent_rec*100:.1f}%, F1={intent_f1*100:.1f}%",
        f"Acc={loc_acc*100:.1f}%, F1={loc_f1*100:.1f}%",
    ],
    "Top-1 (%)": [
        round(intent_acc*100, 2),
        round(loc_acc*100, 2),
    ],
    "Top-3 (%)": ["N/A", "N/A"],
})
results_df = pd.concat([results_df, new_rows], ignore_index=True)
results_df.to_csv(results_path, index=False)

print(f"""
  ── Summary ──
  Intent Classifier    : Acc={intent_acc*100:.1f}%  F1={intent_f1*100:.1f}%
  Location Classifier  : Acc={loc_acc*100:.1f}%  F1={loc_f1*100:.1f}%
  Models saved to      : {MODEL_DIR}

[✓] DistilBERT training complete.
""")
