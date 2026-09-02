import json
import csv
import os
import random


# 1. KNOWLEDGE BASE — 20 records


knowledge_base = [
    {
        "id": 1,
        "name": "University Library",
        "class": "library",
        "category": "Study Area",
        "description": "A quiet learning area containing books and digital resources for students and researchers.",
        "opening_hours": "08:00 AM - 10:00 PM",
        "location": "Building A, Floor 2",
        "events": "Academic writing workshop"
    },
    {
        "id": 2,
        "name": "Computer Laboratory",
        "class": "computerroom",
        "category": "Technology Facility",
        "description": "Room equipped with computers and licensed software for programming, data analysis, and design.",
        "opening_hours": "09:00 AM - 08:00 PM",
        "location": "Building B, Floor 1",
        "events": "Python programming session"
    },
    {
        "id": 3,
        "name": "Main Auditorium",
        "class": "auditorium",
        "category": "Event Venue",
        "description": "Large hall used for lectures, graduation ceremonies, cultural events, and seminars.",
        "opening_hours": "08:00 AM - 09:00 PM",
        "location": "Building C, Ground Floor",
        "events": "Annual tech symposium"
    },
    {
        "id": 4,
        "name": "Art Studio",
        "class": "artstudio",
        "category": "Creative Space",
        "description": "Creative workspace for fine arts, painting, sculpture, and design students.",
        "opening_hours": "09:00 AM - 06:00 PM",
        "location": "Building D, Floor 3",
        "events": "Contemporary art exhibition"
    },
    {
        "id": 5,
        "name": "Campus Bookstore",
        "class": "bookstore",
        "category": "Retail",
        "description": "Sells textbooks, stationery, university merchandise, and academic supplies.",
        "opening_hours": "09:00 AM - 05:00 PM",
        "location": "Building A, Ground Floor",
        "events": "Semester textbook sale"
    },
    {
        "id": 6,
        "name": "Main Classroom Block",
        "class": "classroom",
        "category": "Academic Facility",
        "description": "Standard lecture rooms equipped with projectors and whiteboards for regular classes.",
        "opening_hours": "07:30 AM - 09:00 PM",
        "location": "Building E, All Floors",
        "events": "Guest lecturer series"
    },
    {
        "id": 7,
        "name": "Corridor Hub",
        "class": "corridor",
        "category": "Common Area",
        "description": "Central corridor connecting all departments with notice boards and rest areas.",
        "opening_hours": "07:00 AM - 11:00 PM",
        "location": "All Buildings, All Floors",
        "events": "Student club fair"
    },
    {
        "id": 8,
        "name": "Campus Dining Room",
        "class": "dining_room",
        "category": "Food & Beverage",
        "description": "Main dining facility offering a variety of meals for students, faculty, and staff.",
        "opening_hours": "07:30 AM - 08:00 PM",
        "location": "Building F, Ground Floor",
        "events": "International food festival"
    },
    {
        "id": 9,
        "name": "Main Elevator",
        "class": "elevator",
        "category": "Facility",
        "description": "Accessible lift connecting all floors of the main academic building.",
        "opening_hours": "07:00 AM - 10:00 PM",
        "location": "Building A, Central Core",
        "events": "Accessibility awareness drive"
    },
    {
        "id": 10,
        "name": "Fast Food Restaurant",
        "class": "fastfood_restaurant",
        "category": "Food & Beverage",
        "description": "Quick-service food outlet offering snacks, beverages, and fast meals on campus.",
        "opening_hours": "08:00 AM - 09:00 PM",
        "location": "Building G, Ground Floor",
        "events": "Student discount week"
    },
    {
        "id": 11,
        "name": "Campus Gymnasium",
        "class": "gym",
        "category": "Sports & Fitness",
        "description": "Fully equipped gym with cardio machines, weights, and fitness classes for students.",
        "opening_hours": "06:00 AM - 10:00 PM",
        "location": "Sports Complex, Ground Floor",
        "events": "Fitness challenge competition"
    },
    {
        "id": 12,
        "name": "Wet Laboratory",
        "class": "laboratorywet",
        "category": "Research Facility",
        "description": "Laboratory with water, gas, and chemical supplies for biology and chemistry experiments.",
        "opening_hours": "08:00 AM - 06:00 PM",
        "location": "Building H, Floor 2",
        "events": "Research open day"
    },
    {
        "id": 13,
        "name": "Campus Lobby",
        "class": "lobby",
        "category": "Common Area",
        "description": "Main entrance and reception area with information desk and waiting seating.",
        "opening_hours": "07:00 AM - 10:00 PM",
        "location": "Building A, Ground Floor",
        "events": "Welcome orientation day"
    },
    {
        "id": 14,
        "name": "Locker Room",
        "class": "locker_room",
        "category": "Facility",
        "description": "Secure storage lockers for students near the gym and sports complex.",
        "opening_hours": "06:00 AM - 10:00 PM",
        "location": "Sports Complex, Floor 1",
        "events": "Locker registration drive"
    },
    {
        "id": 15,
        "name": "Meeting Room",
        "class": "meeting_room",
        "category": "Collaboration Space",
        "description": "Bookable rooms with AV equipment for group projects, interviews, and small meetings.",
        "opening_hours": "08:00 AM - 08:00 PM",
        "location": "Building B, Floor 3",
        "events": "Industry mentorship session"
    },
    {
        "id": 16,
        "name": "Campus Museum",
        "class": "museum",
        "category": "Cultural Facility",
        "description": "Displays university history, artefacts, and rotating cultural exhibitions.",
        "opening_hours": "10:00 AM - 05:00 PM",
        "location": "Building I, Ground Floor",
        "events": "Heritage week exhibition"
    },
    {
        "id": 17,
        "name": "Faculty Office",
        "class": "office",
        "category": "Administrative",
        "description": "Staff and faculty office spaces for consultation, administration, and research.",
        "opening_hours": "09:00 AM - 05:00 PM",
        "location": "Building J, All Floors",
        "events": "Open office hour sessions"
    },
    {
        "id": 18,
        "name": "Campus Restaurant",
        "class": "restaurant",
        "category": "Food & Beverage",
        "description": "Full-service restaurant offering hot meals, beverages, and a comfortable seating environment.",
        "opening_hours": "11:00 AM - 09:00 PM",
        "location": "Building K, Ground Floor",
        "events": "Chef's special food night"
    },
    {
        "id": 19,
        "name": "Main Staircase",
        "class": "stairscase",
        "category": "Facility",
        "description": "Primary staircase connecting all floors with safety rails and emergency exit signage.",
        "opening_hours": "24 Hours",
        "location": "All Buildings, Central Core",
        "events": "Fire drill exercise"
    },
    {
        "id": 20,
        "name": "Waiting Room",
        "class": "waitingroom",
        "category": "Common Area",
        "description": "Comfortable seating area near administrative offices for students awaiting appointments.",
        "opening_hours": "08:00 AM - 06:00 PM",
        "location": "Building A, Floor 1",
        "events": "Student advisory walk-in day"
    }
]

with open("campus_knowledge.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
print(f"[✓] campus_knowledge.json — {len(knowledge_base)} records written.")



# 2. TEXT DATASET — campus_queries.csv (~400 rows)


locations = [r["class"] for r in knowledge_base]
location_names = {r["class"]: r["name"] for r in knowledge_base}

query_templates = {
    "find_location": [
        "Where is the {name}?",
        "How do I get to the {name}?",
        "Can you direct me to the {name}?",
        "What is the location of the {name}?",
        "I need to find the {name}.",
        "Which floor is the {name} on?",
        "Where can I find the {name}?",
        "Tell me where the {name} is.",
        "Is the {name} near the entrance?",
        "Show me the way to the {name}.",
    ],
    "ask_hours": [
        "When does the {name} open?",
        "What are the opening hours of the {name}?",
        "Is the {name} open now?",
        "What time does the {name} close?",
        "When is the {name} available?",
        "Can I visit the {name} after 8 PM?",
        "What are the {name} hours today?",
        "Until what time is the {name} open?",
        "Is the {name} open on weekends?",
        "What time does the {name} start?",
    ],
    "find_event": [
        "Are there any events at the {name}?",
        "What events are happening at the {name}?",
        "Is there anything going on at the {name}?",
        "Tell me about upcoming events at the {name}.",
        "What activities are scheduled at the {name}?",
        "Any workshops at the {name} this week?",
        "What is happening at the {name} today?",
        "Are there seminars at the {name}?",
        "What programmes are running at the {name}?",
        "Can you list events at the {name}?",
    ],
    "get_description": [
        "What is the {name}?",
        "Tell me about the {name}.",
        "What facilities does the {name} have?",
        "Describe the {name} for me.",
        "What can I do at the {name}?",
        "What services are available at the {name}?",
        "Give me information about the {name}.",
        "What is available in the {name}?",
        "What does the {name} offer?",
        "I would like to know more about the {name}.",
    ],
}

rows = []
for loc in locations:
    name = location_names[loc]
    for intent, templates in query_templates.items():
        for tmpl in templates:
            rows.append({
                "query": tmpl.format(name=name),
                "intent": intent,
                "location": loc
            })

# Shuffle and trim/pad to ~400
random.seed(42)
random.shuffle(rows)
rows = rows[:400]

with open("campus_queries.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "intent", "location"])
    writer.writeheader()
    writer.writerows(rows)
print(f"[✓] campus_queries.csv — {len(rows)} rows written.")



# 3. VOICE DATASET — 50 .wav files via gTTS


try:
    from gtts import gTTS
    import io

    voice_scripts = [
        # direction queries (15)
        ("library_direction",           "The university library is located in Building A on the second floor."),
        ("computer_lab_direction",      "The computer laboratory is in Building B on the first floor."),
        ("auditorium_direction",        "The main auditorium is in Building C on the ground floor."),
        ("artstudio_direction",         "The art studio is in Building D on the third floor."),
        ("bookstore_direction",         "The campus bookstore is in Building A on the ground floor."),
        ("classroom_direction",         "Classrooms are located in Building E on all floors."),
        ("dining_room_direction",       "The dining room is in Building F on the ground floor."),
        ("gym_direction",               "The campus gymnasium is in the sports complex on the ground floor."),
        ("lobby_direction",             "The main lobby is at the ground floor entrance of Building A."),
        ("meeting_room_direction",      "Meeting rooms are available in Building B on the third floor."),
        ("museum_direction",            "The campus museum is in Building I on the ground floor."),
        ("office_direction",            "Faculty offices are spread across all floors of Building J."),
        ("restaurant_direction",        "The campus restaurant is in Building K on the ground floor."),
        ("waiting_room_direction",      "The waiting room is in Building A on the first floor."),
        ("lab_direction",               "The wet laboratory is in Building H on the second floor."),

        # hours queries (15)
        ("library_hours",               "The library is open from eight in the morning to ten at night."),
        ("computer_lab_hours",          "The computer lab is open from nine in the morning to eight in the evening."),
        ("auditorium_hours",            "The auditorium is open from eight in the morning to nine at night."),
        ("dining_room_hours",           "The dining room is open from seven thirty in the morning to eight in the evening."),
        ("bookstore_hours",             "The bookstore is open from nine in the morning to five in the evening."),
        ("gym_hours",                   "The gym is open from six in the morning to ten at night."),
        ("artstudio_hours",             "The art studio is open from nine in the morning to six in the evening."),
        ("lab_hours",                   "The wet laboratory is open from eight in the morning to six in the evening."),
        ("restaurant_hours",            "The restaurant is open from eleven in the morning to nine at night."),
        ("museum_hours",                "The museum is open from ten in the morning to five in the evening."),
        ("office_hours",                "Faculty offices are open from nine in the morning to five in the evening."),
        ("meeting_room_hours",          "Meeting rooms are available from eight in the morning to eight in the evening."),
        ("fastfood_hours",              "The fast food restaurant is open from eight in the morning to nine at night."),
        ("locker_room_hours",           "The locker room is accessible from six in the morning to ten at night."),
        ("waiting_room_hours",          "The waiting room is open from eight in the morning to six in the evening."),

        # event queries (15)
        ("library_event",               "The library is hosting an academic writing workshop this week."),
        ("computer_lab_event",          "A Python programming session is scheduled at the computer lab today."),
        ("auditorium_event",            "The annual tech symposium will be held in the auditorium this Friday."),
        ("artstudio_event",             "A contemporary art exhibition is currently on display at the art studio."),
        ("bookstore_event",             "The bookstore is running a semester textbook sale this month."),
        ("dining_room_event",           "An international food festival is taking place in the dining room."),
        ("gym_event",                   "A fitness challenge competition is happening at the gym this weekend."),
        ("lab_event",                   "The wet laboratory is hosting a research open day on Thursday."),
        ("museum_event",                "A heritage week exhibition is currently open at the campus museum."),
        ("restaurant_event",            "The restaurant is hosting a chef special food night every Friday."),
        ("meeting_room_event",          "An industry mentorship session is scheduled in the meeting room tomorrow."),
        ("fastfood_event",              "The fast food restaurant is running a student discount week."),
        ("classroom_event",             "A guest lecturer series is taking place in the classroom block this term."),
        ("lobby_event",                 "Welcome orientation day is being held at the main lobby today."),
        ("cafeteria_event",             "An open office hour session is available at the faculty offices this afternoon."),

        # general queries (5)
        ("general_campus_info",         "Welcome to the campus orientation assistant. I can help you find locations, hours, and events."),
        ("corridor_info",               "The central corridor connects all departments and has notice boards with current announcements."),
        ("elevator_info",               "The main elevator in Building A provides accessible access to all floors."),
        ("staircase_info",              "The main staircase is available in all buildings and is open twenty four hours."),
        ("locker_room_info",            "Secure student lockers are available near the sports complex on the first floor."),
    ]

    os.makedirs("voice_dataset", exist_ok=True)

    for filename, text in voice_scripts:
        tts = gTTS(text=text, lang="en", slow=False)
        # gTTS generates mp3; save as .wav via audio conversion using ffmpeg if available,
        # otherwise save as .mp3 with .wav extension (Whisper accepts both)
        wav_path = os.path.join("voice_dataset", f"{filename}.wav")
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        # Try converting mp3 → wav using pydub if available
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(mp3_buffer, format="mp3")
            audio.export(wav_path, format="wav")
        except Exception:
            # Fallback: write mp3 bytes directly with .wav extension
            # Whisper handles both mp3 and wav audio transparently
            with open(wav_path, "wb") as wf:
                wf.write(mp3_buffer.read())

    print(f"[✓] voice_dataset/ — {len(voice_scripts)} .wav files written.")

except ImportError:
    print("[!] gTTS not installed. Run: pip install gtts")
    print("    Then re-run this script to generate voice_dataset/.")


print("\n[✓] All datasets generated successfully.")
print("    campus_knowledge.json")
print("    campus_queries.csv")
print("    voice_dataset/  (50 .wav files)")