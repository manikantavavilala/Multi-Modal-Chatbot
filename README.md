# 🏫 Multimodal Campus Orientation Assistant

A multimodal AI system that helps students and visitors navigate a university campus using **image recognition**, **voice queries**, and **text-based natural language processing**.

## 🏗️ Architecture

```
User Input (Image / Voice / Text)
         │
         ├── 📸 Image  →  CLIP ViT-B/32  →  FAISS Index  →  KB Match
         │
         ├── 🎤 Voice  →  Whisper (base)  →  Transcript  ─┐
         │                                                  │
         ├── ⌨️  Text  ─────────────────────────────────────┤
         │                                                  ▼
         │                                          DistilBERT
         │                                    (Intent + Location)
         │                                                  │
         └──────────────── Fusion MLP ◄─────────────────────┘
                              │
                              ▼
                     Knowledge Base (20 locations)
                              │
                              ▼
                     📋 Response (name, hours, events, directions)
```

### Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Vision | CLIP ViT-B/32 (frozen) + FAISS | Zero-shot image → location retrieval |
| Speech | Whisper base (frozen) | Audio → text transcription |
| NLP | DistilBERT (fine-tuned) | Intent classification + location entity extraction |
| Fusion | Custom MLP with gating | Combines modality embeddings → KB record prediction |
| UI | Streamlit | Web interface with 3 input modes |

---

## 📁 Project Structure

```
Multimodal_Campus_AI_Assistant/
│
├── dataset/
│   ├── image_dataset/          # 20 class folders with campus photos
│   ├── audio_dataset/          # 50 .wav voice query files
│   └── text_dataset/           # campus_queries.csv (400 queries)
│
├── knowledge_base/
│   └── campus_knowledge.json   # 20 location records
│
├── src/
│   ├── 01_data_exploration.py
│   ├── 02_image_preprocessing.py
│   ├── 03_train_clip_faiss.py
│   ├── 04_audio_processing_whisper.py
│   ├── 05_train_distilbert.py
│   ├── 06_train_fusion_model.py
│   └── 07_evaluate_system.py
│
├── saved_models/               # Created during training
│   ├── clip/
│   ├── faiss/
│   ├── distilbert/
│   └── fusion/
│
├── outputs/
│   ├── graphs/                 # All visualisation plots
│   └── results.csv             # Consolidated metrics
│
├── streamlit_app/
│   ├── app.py                  # Main Streamlit UI
│   └── modules/
│       ├── image_prediction.py
│       ├── voice_prediction.py
│       └── chatbot.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 How to Run

### 1. Environment Setup

```bash
# Clone or navigate to the project
cd Multimodal_Campus_AI_Assistant

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training Pipeline (in order)

Each script must be run from the **project root directory**:

```bash
# Step 1: Data Exploration — generates all exploration plots
python src/01_data_exploration.py

# Step 2: Image Preprocessing — demonstrates the augmentation pipeline
python src/02_image_preprocessing.py

# Step 3: CLIP + FAISS — builds image embeddings and search index
python src/03_train_clip_faiss.py

# Step 4: Audio Processing — extracts MFCCs, runs Whisper, computes WER
python src/04_audio_processing_whisper.py

# Step 5: DistilBERT Training — fine-tunes intent and location classifiers
python src/05_train_distilbert.py

# Step 6: Fusion Model — trains the multimodal fusion MLP
python src/06_train_fusion_model.py

# Step 7: Evaluation — runs end-to-end evaluation and generates reports
python src/07_evaluate_system.py
```

### 3. Launch the Web Application

```bash
streamlit run streamlit_app/app.py
```

Then open **http://localhost:8501** in your browser.

### 4. Docker Deployment

```bash
# Build the container
docker build -t campus-assistant .

# Run the container
docker run -p 8501:8501 campus-assistant

# Access at http://localhost:8501
```

### 5. Optional: Expose with Ngrok

```bash
# Install ngrok and authenticate
ngrok http 8501
```

---

## 📊 Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| **Image** | ~2000+ images, 20 classes | Indoor scene photos mapped to campus locations |
| **Audio** | 50 `.wav` files | Synthetic voice queries (directions, hours, events) |
| **Text** | 400 queries | 4 intents × 20 locations, template-generated |
| **Knowledge Base** | 20 records | Name, category, description, hours, location, events |

### Knowledge Base Schema

```json
{
  "id": 1,
  "name": "University Library",
  "class": "library",
  "category": "Study Area",
  "description": "A quiet learning area...",
  "opening_hours": "08:00 AM - 10:00 PM",
  "location": "Building A, Floor 2",
  "events": "Academic writing workshop"
}
```

The `class` field maps directly to the image dataset folder names, enabling end-to-end linking between visual recognition and information retrieval.

---

## 🧪 Test Scenarios

| # | Modality | Input | Expected Output |
|---|----------|-------|-----------------|
| 1 | Image | Upload a library photo | Match → "University Library", Building A Floor 2 |
| 2 | Voice | "Where is the student gym?" | Transcript → intent: find_location → Gymnasium info |
| 3 | Text | "What time does the bookstore close?" | Intent: ask_hours → "09:00 AM - 05:00 PM" |
| 4 | Text | "Any events at the auditorium?" | Intent: find_event → "Annual tech symposium" |
| 5 | Combined | Library image + "What are the hours?" | Fusion → Library + ask_hours → hours info |

---

## 📈 Evaluation Metrics

| Component | Metric | Expected |
|-----------|--------|----------|
| CLIP + FAISS | Top-1 / Top-3 Retrieval Accuracy | Varies by dataset |
| Whisper | Mean Word Error Rate (WER) | <15% on synthetic data |
| DistilBERT (Intent) | Accuracy / F1 | >90% |
| DistilBERT (Location) | Accuracy / F1 | >85% |
| Fusion MLP | End-to-end KB Accuracy | >80% |

---

## ⚖️ Ethical Considerations

- **Data Privacy (GDPR)**: No personal identifiers are collected. Voice data is processed locally using Whisper — no cloud APIs. Images are not stored beyond the session.
- **Bias Mitigation**: Image dataset may over-represent certain building types. Text corpus covers diverse phrasings. Whisper WER may vary across accents.
- **Responsible AI**: System provides informational responses only. Confidence scores are displayed to indicate uncertainty.

---

## 📦 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (optional, CPU works but is slower)
- ~4GB disk space for models
- Docker (optional, for containerisation)

---

## 📄 License

This project is created for academic assessment purposes.
