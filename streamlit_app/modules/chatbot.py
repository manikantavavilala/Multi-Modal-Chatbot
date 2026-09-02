"""
chatbot.py
==========
Streamlit module for text-based campus query processing and KB lookup.

Uses fine-tuned DistilBERT models for intent classification and
location entity extraction, then retrieves the matching knowledge-base
record. Also provides fusion model integration.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertModel,
)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PATH    = os.path.join(ROOT, "knowledge_base", "campus_knowledge.json")
BERT_DIR   = os.path.join(ROOT, "saved_models", "distilbert")
FUSION_DIR = os.path.join(ROOT, "saved_models", "fusion")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── globals ──────────────────────────────────────────────────────────────────
_tokenizer     = None
_intent_model  = None
_loc_model     = None
_bert_base     = None
_fusion_model  = None
_kb            = None
_label_maps    = None


# ═══════════════════════════════════════════════════════════════
# FUSION MLP (must match training definition)
# ═══════════════════════════════════════════════════════════════

class FusionMLP(nn.Module):
    def __init__(self, clip_dim=512, bert_dim=768, num_classes=20):
        super().__init__()
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


# ═══════════════════════════════════════════════════════════════
# LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _load_kb():
    global _kb
    if _kb is None:
        with open(KB_PATH, encoding="utf-8") as f:
            _kb = json.load(f)
    return _kb


def _load_label_maps():
    global _label_maps
    if _label_maps is None:
        map_path = os.path.join(BERT_DIR, "label_maps.json")
        with open(map_path) as f:
            _label_maps = json.load(f)
    return _label_maps


def _load_intent_model():
    global _tokenizer, _intent_model
    if _intent_model is None:
        path = os.path.join(BERT_DIR, "intent")
        _tokenizer = DistilBertTokenizer.from_pretrained(path)
        _intent_model = DistilBertForSequenceClassification.from_pretrained(path).to(DEVICE)
        _intent_model.eval()
    return _tokenizer, _intent_model


def _load_location_model():
    global _loc_model
    if _loc_model is None:
        path = os.path.join(BERT_DIR, "location")
        _loc_model = DistilBertForSequenceClassification.from_pretrained(path).to(DEVICE)
        _loc_model.eval()
    return _loc_model


def _load_bert_base():
    global _bert_base
    if _bert_base is None:
        _bert_base = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
        _bert_base.eval()
    return _bert_base


def _load_fusion_model():
    global _fusion_model
    if _fusion_model is None:
        path = os.path.join(FUSION_DIR, "fusion_model.pt")
        if os.path.exists(path):
            _fusion_model = FusionMLP().to(DEVICE)
            _fusion_model.load_state_dict(
                torch.load(path, map_location=DEVICE, weights_only=True)
            )
            _fusion_model.eval()
    return _fusion_model


# ═══════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def predict_intent(text):
    """
    Classify the intent of a text query.

    Returns:
        dict with:
            - intent: str (e.g. "find_location")
            - confidence: float
            - all_scores: dict of intent → probability
    """
    tokenizer, model = _load_intent_model()
    label_maps = _load_label_maps()
    id2intent = {int(k): v for k, v in label_maps["id2intent"].items()}

    tokens = tokenizer(
        text.lower().strip(), return_tensors="pt",
        max_length=64, padding="max_length", truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**tokens).logits
        probs  = torch.softmax(logits, dim=-1)[0]

    pred_id    = probs.argmax().item()
    confidence = probs[pred_id].item()

    all_scores = {id2intent[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "intent": id2intent[pred_id],
        "confidence": confidence,
        "all_scores": all_scores,
    }


def predict_location(text):
    """
    Predict the campus location entity from a text query.

    Returns:
        dict with:
            - location: str (class name, e.g. "library")
            - location_name: str (display name, e.g. "University Library")
            - confidence: float
    """
    tokenizer, _ = _load_intent_model()
    loc_model = _load_location_model()
    label_maps = _load_label_maps()
    id2loc = {int(k): v for k, v in label_maps["id2loc"].items()}
    kb = _load_kb()

    tokens = tokenizer(
        text.lower().strip(), return_tensors="pt",
        max_length=64, padding="max_length", truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        logits = loc_model(**tokens).logits
        probs  = torch.softmax(logits, dim=-1)[0]

    pred_id    = probs.argmax().item()
    confidence = probs[pred_id].item()
    loc_class  = id2loc[pred_id]

    # Find display name
    loc_name = loc_class
    for record in kb:
        if record["class"] == loc_class:
            loc_name = record["name"]
            break

    return {
        "location": loc_class,
        "location_name": loc_name,
        "confidence": confidence,
    }


def lookup_knowledge_base(location_class):
    """
    Retrieve the full KB record for a given location class.

    Returns:
        dict — the KB record, or None if not found
    """
    kb = _load_kb()
    for record in kb:
        if record["class"] == location_class:
            return record
    return None


def get_text_embedding(text):
    """
    Get the DistilBERT [CLS] embedding for a text query (for fusion model).

    Returns:
        numpy array of shape (768,)
    """
    tokenizer, _ = _load_intent_model()
    bert_base = _load_bert_base()

    tokens = tokenizer(
        text.lower().strip(), return_tensors="pt",
        max_length=64, padding="max_length", truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bert_base(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0, :]

    return cls_emb.cpu().numpy().flatten()


def get_fusion_prediction(image_emb=None, text_emb=None):
    """
    Get the fusion model's prediction given one or both modality embeddings.

    Args:
        image_emb: numpy array (512,) or None
        text_emb:  numpy array (768,) or None

    Returns:
        dict with:
            - record: KB record dict
            - confidence: float
            - class_name: str
    """
    fusion = _load_fusion_model()
    kb = _load_kb()

    if fusion is None:
        return None

    # Prepare inputs
    if image_emb is not None:
        img_t = torch.tensor(image_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        img_m = torch.ones(1, 1).to(DEVICE)
    else:
        img_t = torch.zeros(1, 512).to(DEVICE)
        img_m = torch.zeros(1, 1).to(DEVICE)

    if text_emb is not None:
        txt_t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        txt_m = torch.ones(1, 1).to(DEVICE)
    else:
        txt_t = torch.zeros(1, 768).to(DEVICE)
        txt_m = torch.zeros(1, 1).to(DEVICE)

    with torch.no_grad():
        logits = fusion(img_t, txt_t, img_m, txt_m)
        probs  = torch.softmax(logits, dim=-1)[0]

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()

    if pred_idx < len(kb):
        record = kb[pred_idx]
    else:
        record = None

    return {
        "record": record,
        "confidence": confidence,
        "class_name": kb[pred_idx]["class"] if record else "unknown",
    }


def format_response(record, intent="get_description"):
    """
    Format a KB record into a user-friendly response based on intent.

    Returns:
        str — formatted response text
    """
    if record is None:
        return "I'm sorry, I couldn't find information about that location."

    name     = record.get("name", "Unknown")
    desc     = record.get("description", "No description available.")
    hours    = record.get("opening_hours", "Not specified")
    location = record.get("location", "Not specified")
    events   = record.get("events", "No events scheduled")
    category = record.get("category", "")

    if intent == "find_location":
        return (
            f"📍 **{name}**\n\n"
            f"**Location:** {location}\n\n"
            f"**Category:** {category}\n\n"
            f"**Description:** {desc}"
        )
    elif intent == "ask_hours":
        return (
            f"🕐 **{name}**\n\n"
            f"**Opening Hours:** {hours}\n\n"
            f"**Location:** {location}"
        )
    elif intent == "find_event":
        return (
            f"🎉 **{name}**\n\n"
            f"**Upcoming Events:** {events}\n\n"
            f"**Location:** {location}\n\n"
            f"**Hours:** {hours}"
        )
    else:  # get_description
        return (
            f"ℹ️ **{name}**\n\n"
            f"**Category:** {category}\n\n"
            f"**Description:** {desc}\n\n"
            f"**Location:** {location}\n\n"
            f"**Opening Hours:** {hours}\n\n"
            f"**Events:** {events}"
        )
