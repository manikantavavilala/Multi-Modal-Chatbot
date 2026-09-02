"""
image_prediction.py
===================
Streamlit module for image-based campus location prediction.

Uses CLIP (ViT-B/32) to encode an uploaded image and FAISS to find the
nearest knowledge-base text description via cosine similarity.
"""

import os
import json
import pickle
import numpy as np
import torch
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PATH   = os.path.join(ROOT, "knowledge_base", "campus_knowledge.json")
CLIP_DIR  = os.path.join(ROOT, "saved_models", "clip")
FAISS_DIR = os.path.join(ROOT, "saved_models", "faiss")

# ── globals (loaded once) ────────────────────────────────────────────────────
_clip_model    = None
_clip_preprocess = None
_faiss_index   = None
_kb            = None
_text_embeddings = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_models():
    """Lazy-load CLIP model, FAISS index, and knowledge base."""
    global _clip_model, _clip_preprocess, _faiss_index, _kb, _text_embeddings

    if _clip_model is None:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        _clip_model.eval()

    if _faiss_index is None:
        import faiss
        _faiss_index = faiss.read_index(os.path.join(FAISS_DIR, "campus_index.faiss"))

    if _kb is None:
        with open(KB_PATH, encoding="utf-8") as f:
            _kb = json.load(f)

    if _text_embeddings is None:
        with open(os.path.join(CLIP_DIR, "text_embeddings.pkl"), "rb") as f:
            _text_embeddings = pickle.load(f)


def predict_from_image(image, top_k=3):
    """
    Predict campus location from a PIL Image.

    Args:
        image: PIL.Image — the uploaded campus photo
        top_k: number of top matches to return

    Returns:
        list of dicts, each containing:
            - record: full KB record dict
            - score: cosine similarity score
    """
    _load_models()

    # Preprocess and encode
    image_input = _clip_preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_embedding = _clip_model.encode_image(image_input)
        image_embedding = image_embedding.cpu().numpy().astype("float32")
        image_embedding = image_embedding / np.linalg.norm(image_embedding)

    # Search FAISS index
    distances, indices = _faiss_index.search(image_embedding, top_k)

    results = []
    for rank in range(top_k):
        kb_idx = indices[0][rank]
        score  = float(distances[0][rank])
        if 0 <= kb_idx < len(_kb):
            results.append({
                "record": _kb[kb_idx],
                "score": score,
                "rank": rank + 1,
            })

    return results


def get_image_embedding(image):
    """
    Get the raw CLIP embedding for an image (for fusion model).

    Args:
        image: PIL.Image

    Returns:
        numpy array of shape (512,)
    """
    _load_models()

    image_input = _clip_preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _clip_model.encode_image(image_input)
        emb = emb.cpu().numpy().astype("float32")
        emb = emb / np.linalg.norm(emb)

    return emb.flatten()
