"""
voice_prediction.py
===================
Streamlit module for voice-based campus query processing.

Uses Whisper (base) to transcribe audio, then passes the transcript
to the text pipeline for intent classification and KB lookup.
"""

import os
import tempfile
import numpy as np

# ── globals ──────────────────────────────────────────────────────────────────
_whisper_model = None


def _load_whisper():
    """Lazy-load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_audio(audio_path):
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path: str — path to the audio file (.wav, .mp3, etc.)

    Returns:
        dict with keys:
            - text: transcribed text
            - language: detected language
            - segments: list of timestamped segments
    """
    model = _load_whisper()
    result = model.transcribe(audio_path, language="en")

    return {
        "text": result["text"].strip(),
        "language": result.get("language", "en"),
        "segments": result.get("segments", []),
    }


def transcribe_audio_bytes(audio_bytes, suffix=".wav"):
    """
    Transcribe audio from bytes (e.g., from Streamlit file uploader).

    Args:
        audio_bytes: bytes — raw audio data
        suffix: str — file extension hint

    Returns:
        dict with transcription result (same as transcribe_audio)
    """
    # Write to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result


def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    """
    Extract MFCC features from an audio file (for display/analysis).

    Args:
        audio_path: str — path to audio file
        sr: sample rate
        n_mfcc: number of MFCC coefficients

    Returns:
        numpy array of shape (n_mfcc, time_frames)
    """
    import librosa
    y, sr_loaded = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc
