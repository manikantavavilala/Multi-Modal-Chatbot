import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH   = os.path.join(ROOT, "knowledge_base", "campus_knowledge.json")
AUD_DIR   = os.path.join(ROOT, "dataset", "audio_dataset")
GRAPH_DIR = os.path.join(ROOT, "outputs", "graphs")
OUT_DIR   = os.path.join(ROOT, "outputs")

for d in [GRAPH_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1.  AUDIO FEATURE EXTRACTION (MFCC)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  1.  AUDIO FEATURE EXTRACTION (MFCC)")
print("="*70)

import librosa
import librosa.display

audio_files = sorted([f for f in os.listdir(AUD_DIR) if f.endswith(".wav")])
print(f"\n  Total audio files: {len(audio_files)}")

# Extract MFCCs for all files
all_mfccs     = []
all_durations = []
file_names    = []

for af in tqdm(audio_files, desc="  Extracting MFCCs"):
    filepath = os.path.join(AUD_DIR, af)
    y, sr = librosa.load(filepath, sr=16000)  # resample to 16kHz for Whisper
    duration = librosa.get_duration(y=y, sr=sr)

    # 13 MFCC coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    all_mfccs.append(mfcc)
    all_durations.append(duration)
    file_names.append(af)

print(f"  MFCC shape (sample) : {all_mfccs[0].shape}  (13 coefficients × time frames)")
print(f"  Average duration    : {np.mean(all_durations):.2f}s")

# ── 1a. Normalise MFCCs ─────────────────────────────────────
# Compute global mean and std across all MFCC frames
all_frames = np.concatenate([m for m in all_mfccs], axis=1)
global_mean = all_frames.mean(axis=1, keepdims=True)
global_std  = all_frames.std(axis=1, keepdims=True) + 1e-8

normalised_mfccs = [(m - global_mean) / global_std for m in all_mfccs]
print(f"  Global MFCC mean   : {global_mean.flatten()[:3]}... (first 3 coeffs)")
print(f"  Global MFCC std    : {global_std.flatten()[:3]}... (first 3 coeffs)")

# ── 1b. Pad sequences to same length ────────────────────────
max_len = max(m.shape[1] for m in normalised_mfccs)
padded_mfccs = np.zeros((len(normalised_mfccs), 13, max_len), dtype=np.float32)
for i, m in enumerate(normalised_mfccs):
    padded_mfccs[i, :, :m.shape[1]] = m

print(f"  Padded shape       : {padded_mfccs.shape}  (batch × coeffs × max_frames)")
print("  [✓] MFCCs extracted, normalised, and padded")

# ── 1c. MFCC spectrogram visualisation ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
sample_indices = [0, len(audio_files)//4, len(audio_files)//2,
                  3*len(audio_files)//4, len(audio_files)-2, len(audio_files)-1]
for i, idx in enumerate(sample_indices):
    row, col = i // 3, i % 3
    y_vis, sr_vis = librosa.load(os.path.join(AUD_DIR, audio_files[idx]), sr=16000)
    mfcc_vis = librosa.feature.mfcc(y=y_vis, sr=sr_vis, n_mfcc=13)
    img = librosa.display.specshow(mfcc_vis, x_axis="time", ax=axes[row][col], sr=sr_vis)
    axes[row][col].set_title(audio_files[idx].replace(".wav", ""), fontsize=9)
    axes[row][col].set_ylabel("MFCC" if col == 0 else "")
fig.suptitle("Audio Preprocessing — MFCC Spectrograms", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "mfcc_spectrograms.png"), dpi=150)
plt.close()
print("  [✓] Saved: mfcc_spectrograms.png")


# ═══════════════════════════════════════════════════════════════
# 2.  WHISPER TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  2.  WHISPER ASR TRANSCRIPTION")
print("="*70)

import whisper

print("  Loading Whisper (base) model...")
whisper_model = whisper.load_model("base")
print("  [✓] Whisper loaded")

# Ground-truth transcripts (from generate_datasets.py)
ground_truth = {
    "library_direction":       "The university library is located in Building A on the second floor.",
    "computer_lab_direction":  "The computer laboratory is in Building B on the first floor.",
    "auditorium_direction":    "The main auditorium is in Building C on the ground floor.",
    "artstudio_direction":     "The art studio is in Building D on the third floor.",
    "bookstore_direction":     "The campus bookstore is in Building A on the ground floor.",
    "classroom_direction":     "Classrooms are located in Building E on all floors.",
    "dining_room_direction":   "The dining room is in Building F on the ground floor.",
    "gym_direction":           "The campus gymnasium is in the sports complex on the ground floor.",
    "lobby_direction":         "The main lobby is at the ground floor entrance of Building A.",
    "meeting_room_direction":  "Meeting rooms are available in Building B on the third floor.",
    "museum_direction":        "The campus museum is in Building I on the ground floor.",
    "office_direction":        "Faculty offices are spread across all floors of Building J.",
    "restaurant_direction":    "The campus restaurant is in Building K on the ground floor.",
    "waiting_room_direction":  "The waiting room is in Building A on the first floor.",
    "lab_direction":           "The wet laboratory is in Building H on the second floor.",
    "library_hours":           "The library is open from eight in the morning to ten at night.",
    "computer_lab_hours":      "The computer lab is open from nine in the morning to eight in the evening.",
    "auditorium_hours":        "The auditorium is open from eight in the morning to nine at night.",
    "dining_room_hours":       "The dining room is open from seven thirty in the morning to eight in the evening.",
    "bookstore_hours":         "The bookstore is open from nine in the morning to five in the evening.",
    "gym_hours":               "The gym is open from six in the morning to ten at night.",
    "artstudio_hours":         "The art studio is open from nine in the morning to six in the evening.",
    "lab_hours":               "The wet laboratory is open from eight in the morning to six in the evening.",
    "restaurant_hours":        "The restaurant is open from eleven in the morning to nine at night.",
    "museum_hours":            "The museum is open from ten in the morning to five in the evening.",
    "office_hours":            "Faculty offices are open from nine in the morning to five in the evening.",
    "meeting_room_hours":      "Meeting rooms are available from eight in the morning to eight in the evening.",
    "fastfood_hours":          "The fast food restaurant is open from eight in the morning to nine at night.",
    "locker_room_hours":       "The locker room is accessible from six in the morning to ten at night.",
    "waiting_room_hours":      "The waiting room is open from eight in the morning to six in the evening.",
    "library_event":           "The library is hosting an academic writing workshop this week.",
    "computer_lab_event":      "A Python programming session is scheduled at the computer lab today.",
    "auditorium_event":        "The annual tech symposium will be held in the auditorium this Friday.",
    "artstudio_event":         "A contemporary art exhibition is currently on display at the art studio.",
    "bookstore_event":         "The bookstore is running a semester textbook sale this month.",
    "dining_room_event":       "An international food festival is taking place in the dining room.",
    "gym_event":               "A fitness challenge competition is happening at the gym this weekend.",
    "lab_event":               "The wet laboratory is hosting a research open day on Thursday.",
    "museum_event":            "A heritage week exhibition is currently open at the campus museum.",
    "restaurant_event":        "The restaurant is hosting a chef special food night every Friday.",
    "meeting_room_event":      "An industry mentorship session is scheduled in the meeting room tomorrow.",
    "fastfood_event":          "The fast food restaurant is running a student discount week.",
    "classroom_event":         "A guest lecturer series is taking place in the classroom block this term.",
    "lobby_event":             "Welcome orientation day is being held at the main lobby today.",
    "cafeteria_event":         "An open office hour session is available at the faculty offices this afternoon.",
    "general_campus_info":     "Welcome to the campus orientation assistant. I can help you find locations, hours, and events.",
    "corridor_info":           "The central corridor connects all departments and has notice boards with current announcements.",
    "elevator_info":           "The main elevator in Building A provides accessible access to all floors.",
    "staircase_info":          "The main staircase is available in all buildings and is open twenty four hours.",
    "locker_room_info":        "Secure student lockers are available near the sports complex on the first floor.",
}

# Transcribe all audio files
transcriptions = []
print(f"\n  Transcribing {len(audio_files)} files...")
for af in tqdm(audio_files, desc="  Transcribing"):
    filepath = os.path.join(AUD_DIR, af)
    result   = whisper_model.transcribe(filepath, language="en")
    transcript = result["text"].strip()

    file_key = af.replace(".wav", "")
    gt_text  = ground_truth.get(file_key, "")

    transcriptions.append({
        "filename": af,
        "ground_truth": gt_text,
        "whisper_transcript": transcript
    })

# Save transcriptions
trans_df = pd.DataFrame(transcriptions)
trans_path = os.path.join(OUT_DIR, "whisper_transcriptions.csv")
trans_df.to_csv(trans_path, index=False)
print(f"  [✓] Transcriptions saved to: {trans_path}")

# Print sample transcriptions
print(f"\n  ── Sample Transcriptions ──")
for i in range(min(5, len(transcriptions))):
    t = transcriptions[i]
    print(f"\n    File: {t['filename']}")
    print(f"    GT  : {t['ground_truth'][:80]}...")
    print(f"    ASR : {t['whisper_transcript'][:80]}...")


# ═══════════════════════════════════════════════════════════════
# 3.  WORD ERROR RATE (WER) CALCULATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  3.  WORD ERROR RATE (WER) EVALUATION")
print("="*70)

from jiwer import wer as compute_wer

wer_scores = []
for t in transcriptions:
    if t["ground_truth"]:
        w = compute_wer(t["ground_truth"].lower(), t["whisper_transcript"].lower())
        wer_scores.append(w)
        t["wer"] = w
    else:
        t["wer"] = None

wer_scores = np.array(wer_scores)
mean_wer = wer_scores.mean() * 100
median_wer = np.median(wer_scores) * 100

print(f"\n  Files evaluated  : {len(wer_scores)}")
print(f"  Mean WER         : {mean_wer:.2f}%")
print(f"  Median WER       : {median_wer:.2f}%")
print(f"  Min WER          : {wer_scores.min()*100:.2f}%")
print(f"  Max WER          : {wer_scores.max()*100:.2f}%")
print(f"  Perfect (0% WER) : {(wer_scores == 0).sum()}/{len(wer_scores)}")

# ── 3a. WER distribution plot ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(wer_scores * 100, bins=20, color="#ed7d31", edgecolor="white", alpha=0.85)
ax.axvline(mean_wer, color="red", linestyle="--", label=f"Mean: {mean_wer:.1f}%")
ax.axvline(median_wer, color="blue", linestyle=":", label=f"Median: {median_wer:.1f}%")
ax.set_xlabel("Word Error Rate (%)")
ax.set_ylabel("Number of Files")
ax.set_title("Whisper ASR — Word Error Rate Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "wer_distribution.png"), dpi=150)
plt.close()
print("  [✓] Saved: wer_distribution.png")

# ── 3b. Update results CSV ──────────────────────────────────
results_path = os.path.join(OUT_DIR, "results.csv")
try:
    results_df = pd.read_csv(results_path)
except FileNotFoundError:
    results_df = pd.DataFrame()

new_row = pd.DataFrame({
    "Component": ["Whisper ASR (Speech)"],
    "Metric": ["Word Error Rate"],
    "Top-1 (%)": [round(mean_wer, 2)],
    "Top-3 (%)": ["N/A"],
})
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv(results_path, index=False)
print(f"  [✓] Results updated: {results_path}")

print(f"""
  ── Summary ──
  MFCC Features : 13 coefficients, normalised, padded to {max_len} frames
  Whisper Model : base (frozen)
  Mean WER      : {mean_wer:.2f}%
  Median WER    : {median_wer:.2f}%

[✓] Audio processing and Whisper pipeline complete.
""")
