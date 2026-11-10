import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import app.db.database
from fastapi.testclient import TestClient
from app.main import app

def simulate():
    client = TestClient(app)
    conv = client.post("/api/conversations", json={"title": "Full Realistic CBT Simulation"}).json()
    cid = conv["id"]
    print(f"\n==================================================")
    print(f"🎬 STARTING REALISTIC PATIENT CBT SESSION (ID: {cid})")
    print(f"==================================================\n")

    conversation_steps = [
        ("Patient Greeting", "Hello!"),
        ("Problem Statement", "I am feeling really stressed and overwhelmed about my final year project."),
        ("Direct Patient Question", "Can CBT actually help with stress like this?"),
        ("Situational Context", "My teammate suddenly dropped out yesterday and left all the coding to me."),
        ("Somatic Symptoms", "My chest was tight, my heart was racing, and I couldn't sleep all night."),
        ("Automatic Thought", "I keep thinking I am a total failure and my entire degree is ruined forever."),
        ("Evidence Examination", "I know I got good grades in other semesters, but this one feels completely impossible."),
        ("Solution / Action Step", "What should I do right now to get through today?"),
        ("Explicit Exit / Closure", "Thank you so much for this, I have to go to class now.")
    ]

    for turn_num, (description, user_msg) in enumerate(conversation_steps, 1):
        resp = client.post(f"/api/conversations/{cid}/messages", json={"text": user_msg}).json()
        decision = resp["decision"]
        analysis = resp["analysis"]
        ai_text = resp["ai_message"]["content"]
        distortion = analysis["distortion"]["predicted_class"]

        print(f"--- Turn {turn_num}: {description} ---")
        print(f"👤 Patient   : {user_msg}")
        print(f"🎯 Stage     : {decision['therapy_stage']} (Reason: {decision['reason']})")
        print(f"🧠 Distortion: {distortion}")
        print(f"🤖 Better Me : {ai_text}")
        print()

if __name__ == "__main__":
    simulate()
