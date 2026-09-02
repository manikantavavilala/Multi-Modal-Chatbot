"""
Generate voice QUESTION audio files using gTTS.
These are user-style queries like "Where is the library?"
"""
import os
import io

from gtts import gTTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "audio_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

voice_queries = [
    # find_location queries (15)
    ("q_library_location",        "Where is the university library?"),
    ("q_gym_location",            "How do I get to the campus gymnasium?"),
    ("q_auditorium_location",     "Can you direct me to the main auditorium?"),
    ("q_bookstore_location",      "Where can I find the campus bookstore?"),
    ("q_computerlab_location",    "Which floor is the computer laboratory on?"),
    ("q_dining_location",         "Where is the dining room?"),
    ("q_museum_location",         "I need to find the campus museum."),
    ("q_restaurant_location",     "Show me the way to the campus restaurant."),
    ("q_lab_location",            "What is the location of the wet laboratory?"),
    ("q_office_location",         "Where are the faculty offices?"),
    ("q_lobby_location",          "How do I get to the main lobby?"),
    ("q_meetingroom_location",    "Where is the meeting room?"),
    ("q_artstudio_location",      "Can you tell me where the art studio is?"),
    ("q_elevator_location",       "Where is the main elevator?"),
    ("q_waitingroom_location",    "I need to find the waiting room."),

    # ask_hours queries (10)
    ("q_library_hours",           "What are the opening hours of the library?"),
    ("q_gym_hours",               "What time does the gym open?"),
    ("q_bookstore_hours",         "Is the bookstore open right now?"),
    ("q_museum_hours",            "When does the campus museum close?"),
    ("q_restaurant_hours",        "What time is the restaurant open until?"),
    ("q_computerlab_hours",       "Is the computer lab open on weekends?"),
    ("q_dining_hours",            "What are the dining room hours today?"),
    ("q_office_hours",            "When are the faculty offices open?"),
    ("q_artstudio_hours",         "Until what time is the art studio open?"),
    ("q_fastfood_hours",          "What time does the fast food restaurant close?"),

    # find_event queries (10)
    ("q_library_event",           "Are there any events at the library this week?"),
    ("q_auditorium_event",        "What events are happening at the auditorium?"),
    ("q_gym_event",               "Is there anything going on at the gym?"),
    ("q_museum_event",            "Tell me about upcoming events at the museum."),
    ("q_artstudio_event",         "Any workshops at the art studio this week?"),
    ("q_bookstore_event",         "What activities are scheduled at the bookstore?"),
    ("q_dining_event",            "Is there a food festival in the dining room?"),
    ("q_classroom_event",         "Are there seminars in the classroom block?"),
    ("q_meetingroom_event",       "What programmes are running at the meeting room?"),
    ("q_lab_event",               "Are there any events at the wet laboratory?"),

    # get_description queries (5)
    ("q_library_info",            "Tell me about the university library."),
    ("q_gym_info",                "What facilities does the gymnasium have?"),
    ("q_computerlab_info",        "What can I do at the computer laboratory?"),
    ("q_lobby_info",              "What is the campus lobby?"),
    ("q_locker_info",             "Describe the locker room for me."),
]

print(f"Generating {len(voice_queries)} voice query files...")

for filename, text in voice_queries:
    wav_path = os.path.join(OUT_DIR, f"{filename}.wav")
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(wav_path)
    print(f"  [OK] {filename}.wav")

print(f"\n[OK] Done -- {len(voice_queries)} question audio files generated in {OUT_DIR}")
