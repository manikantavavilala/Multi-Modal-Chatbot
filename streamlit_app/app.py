"""
app.py — Campus Orientation Assistant
======================================
Assessment Section: Deployment & User Testing (10%)

A professional Streamlit web application supporting three input modes:
  1. 📸 Image Upload  — upload a campus photo for location identification
  2. 🎤 Voice Input   — upload an audio query for transcription and processing
  3. ⌨️  Text Query    — type a question about the campus

The output panel displays:
  • Matched campus location name
  • Description and category
  • Opening hours
  • Events
  • Directional information

Run:
    streamlit run streamlit_app/app.py
"""

import os
import sys
import json
import streamlit as st
from PIL import Image

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import modules
from streamlit_app.modules.image_prediction import predict_from_image, get_image_embedding
from streamlit_app.modules.voice_prediction import transcribe_audio_bytes
from streamlit_app.modules.chatbot import (
    predict_intent, predict_location, lookup_knowledge_base,
    get_text_embedding, get_fusion_prediction, format_response,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏫 Campus Orientation Assistant",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }

    /* Result cards */
    .result-card {
        background: rgba(102, 126, 234, 0.08);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: inherit;
    }

    /* Info boxes — theme aware */
    .info-box {
        background: rgba(102, 126, 234, 0.10);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: inherit;
        backdrop-filter: blur(4px);
    }
    .info-box strong {
        color: inherit;
        font-size: 0.95rem;
    }

    /* Confidence bar — theme aware */
    .confidence-bar {
        background: rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        height: 20px;
        margin: 5px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        height: 100%;
        transition: width 0.5s ease;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏫 Campus Orientation Assistant</h1>
    <p>Your AI-powered guide to campus locations, hours, and events.<br>
    Upload an image, record a voice query, or type your question!</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/university.png", width=120)
    st.markdown("## 🧭 Navigation")
    st.markdown("""
    This assistant understands three types of input:

    - 📸 **Image**: Upload a photo of a campus location
    - 🎤 **Voice**: Upload an audio file with your question
    - ⌨️ **Text**: Type your question directly

    The system will identify the location and provide:
    - 📍 Location & directions
    - 🕐 Opening hours
    - 🎉 Upcoming events
    - ℹ️ Facility description
    """)

    st.markdown("---")
    st.markdown("### 🔧 System Info")
    st.markdown(f"""
    - **Vision**: CLIP ViT-B/32 + FAISS
    - **Speech**: OpenAI Whisper (base)
    - **NLP**: DistilBERT (fine-tuned)
    - **Fusion**: Gated MLP
    """)


# ── Helper functions ─────────────────────────────────────────────────────────

def display_result(record, intent="get_description", confidence=None, source=""):
    """Display a KB record as a nicely formatted result card."""
    if record is None:
        st.warning("⚠️ Could not identify the location. Please try again.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### 📍 {record['name']}")
        st.markdown(f"**Category:** {record.get('category', 'N/A')}")
        st.markdown(f"**Description:** {record.get('description', 'N/A')}")

        # Formatted response based on intent
        response = format_response(record, intent)
        st.markdown("---")
        st.markdown(response)

    with col2:
        st.markdown("#### 📋 Quick Info")

        st.markdown(f"""
        <div class="info-box">
            <strong>🕐 Hours</strong><br>
            {record.get('opening_hours', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <strong>📍 Location</strong><br>
            {record.get('location', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <strong>🎉 Events</strong><br>
            {record.get('events', 'No events')}
        </div>
        """, unsafe_allow_html=True)

        if confidence is not None:
            pct = int(confidence * 100)
            st.markdown(f"""
            <div class="info-box">
                <strong>🎯 Confidence: {pct}%</strong>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if source:
            st.caption(f"Source: {source}")


# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image Upload", "🎤 Voice Input", "⌨️ Text Query", "🔗 Combined Input"
])


# ── Tab 1: Image Upload ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📸 Upload a Campus Photo")
    st.markdown("Upload an image of a campus location and the AI will identify it.")

    uploaded_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png", "bmp"],
        key="image_upload"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col_res:
            with st.spinner("🔍 Analysing image with CLIP + FAISS..."):
                results = predict_from_image(image, top_k=3)

            if results:
                st.success(f"✅ Top match: **{results[0]['record']['name']}**")
                display_result(
                    results[0]["record"],
                    intent="get_description",
                    confidence=min(results[0]["score"], 1.0),
                    source="CLIP + FAISS Image Retrieval"
                )

                # Show alternatives
                if len(results) > 1:
                    with st.expander("🔄 Alternative Matches"):
                        for r in results[1:]:
                            st.markdown(f"**{r['rank']}. {r['record']['name']}** "
                                      f"(score: {r['score']:.3f})")
            else:
                st.error("❌ No matches found.")


# ── Tab 2: Voice Input ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🎤 Upload a Voice Query")
    st.markdown("Upload an audio file (.wav, .mp3) with your campus question.")

    uploaded_audio = st.file_uploader(
        "Choose an audio file...", type=["wav", "mp3", "m4a", "ogg"],
        key="audio_upload"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")

        if st.button("🎙️ Transcribe & Process", key="transcribe_btn"):
            with st.spinner("🔊 Transcribing with Whisper..."):
                audio_bytes = uploaded_audio.getvalue()
                suffix = os.path.splitext(uploaded_audio.name)[1]
                transcript_result = transcribe_audio_bytes(audio_bytes, suffix=suffix)

            transcript = transcript_result["text"]

            st.markdown("#### 📝 Transcription")
            st.info(f'"{transcript}"')

            if transcript:
                with st.spinner("🧠 Processing with DistilBERT..."):
                    intent_result = predict_intent(transcript)
                    loc_result    = predict_location(transcript)

                st.markdown(f"**Detected Intent:** `{intent_result['intent']}` "
                          f"(confidence: {intent_result['confidence']:.1%})")
                st.markdown(f"**Detected Location:** `{loc_result['location_name']}` "
                          f"(confidence: {loc_result['confidence']:.1%})")

                record = lookup_knowledge_base(loc_result["location"])
                st.markdown("---")
                display_result(
                    record,
                    intent=intent_result["intent"],
                    confidence=loc_result["confidence"],
                    source="Whisper → DistilBERT"
                )


# ── Tab 3: Text Query ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### ⌨️ Type Your Question")
    st.markdown("Ask about campus locations, hours, events, or facilities.")

    # Example queries
    st.markdown("**Try these examples:**")
    examples = [
        "Where is the University Library?",
        "What are the opening hours of the gym?",
        "Are there any events at the auditorium?",
        "Tell me about the Computer Laboratory.",
    ]

    selected_example = st.selectbox("Quick examples:", [""] + examples)

    text_input = st.text_input(
        "Your question:",
        value=selected_example,
        placeholder="e.g., Where is the library?",
        key="text_input"
    )

    if text_input:
        with st.spinner("🧠 Processing with DistilBERT..."):
            intent_result = predict_intent(text_input)
            loc_result    = predict_location(text_input)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🎯 Intent:** `{intent_result['intent']}`")
            # Intent confidence scores
            for intent_name, score in sorted(intent_result["all_scores"].items(),
                                             key=lambda x: -x[1]):
                st.progress(score, text=f"{intent_name}: {score:.1%}")

        with col2:
            st.markdown(f"**📍 Location:** `{loc_result['location_name']}`")
            st.markdown(f"**Confidence:** {loc_result['confidence']:.1%}")

        record = lookup_knowledge_base(loc_result["location"])
        st.markdown("---")
        display_result(
            record,
            intent=intent_result["intent"],
            confidence=loc_result["confidence"],
            source="DistilBERT Text Classification"
        )


# ── Tab 4: Combined Input ───────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔗 Combined Multimodal Input")
    st.markdown("Provide any combination of inputs for fusion-based prediction.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📸 Image (optional)")
        combined_image = st.file_uploader(
            "Upload image...", type=["jpg", "jpeg", "png"],
            key="combined_image"
        )

        st.markdown("#### ⌨️ Text (optional)")
        combined_text = st.text_input(
            "Your question:", placeholder="e.g., Where is this place?",
            key="combined_text"
        )

    with col_right:
        st.markdown("#### 🎤 Voice (optional)")
        combined_audio = st.file_uploader(
            "Upload audio...", type=["wav", "mp3"],
            key="combined_audio"
        )

    if st.button("🚀 Process with Fusion Model", key="fusion_btn"):
        image_emb = None
        text_emb  = None
        text_query = combined_text or ""

        # Process image
        if combined_image is not None:
            with st.spinner("Processing image..."):
                img = Image.open(combined_image).convert("RGB")
                image_emb = get_image_embedding(img)
                st.image(img, caption="Uploaded Image", width=300)

        # Process audio → get transcript → treat as text
        if combined_audio is not None:
            with st.spinner("Transcribing audio..."):
                audio_bytes = combined_audio.getvalue()
                suffix = os.path.splitext(combined_audio.name)[1]
                transcript = transcribe_audio_bytes(audio_bytes, suffix=suffix)
                voice_text = transcript["text"]
                st.info(f"🎙️ Transcript: \"{voice_text}\"")
                if voice_text:
                    text_query = voice_text if not text_query else f"{text_query} {voice_text}"

        # Get text embedding
        if text_query:
            with st.spinner("Encoding text..."):
                text_emb = get_text_embedding(text_query)
                # Also get intent for response formatting
                intent_result = predict_intent(text_query)
                detected_intent = intent_result["intent"]
        else:
            detected_intent = "get_description"

        # Check if at least one modality is provided
        if image_emb is None and text_emb is None:
            st.warning("⚠️ Please provide at least one input (image, text, or voice).")
        else:
            with st.spinner("🔗 Running fusion model..."):
                fusion_result = get_fusion_prediction(
                    image_emb=image_emb,
                    text_emb=text_emb
                )

            if fusion_result and fusion_result["record"]:
                modalities_used = []
                if image_emb is not None:
                    modalities_used.append("Image")
                if text_emb is not None:
                    modalities_used.append("Text")
                source = f"Fusion MLP ({' + '.join(modalities_used)})"

                st.success(f"✅ Matched: **{fusion_result['record']['name']}**")
                display_result(
                    fusion_result["record"],
                    intent=detected_intent,
                    confidence=fusion_result["confidence"],
                    source=source
                )
            else:
                st.error("❌ Fusion model could not determine the location.")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 1rem;'>"
    "🏫 Campus Orientation Assistant | Multimodal AI System | "
    "CLIP + Whisper + DistilBERT + Fusion MLP"
    "</div>",
    unsafe_allow_html=True
)
