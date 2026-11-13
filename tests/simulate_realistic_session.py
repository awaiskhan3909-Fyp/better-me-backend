import sys
import uuid
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import app.db.database
from fastapi.testclient import TestClient
from app.main import app

# 10 Multi-turn clinical CBT scenarios
SCENARIOS = [
    {
        "id": 1,
        "name": "Exam Stress",
        "description": "Anticipatory anxiety and fear of failure before finals",
        "turns": [
            "I have my final algorithms exam tomorrow morning and I am so anxious I cannot focus on my notes.",
            "I keep thinking that if I fail this exam, I will lose my scholarship and disappoint my entire family.",
            "I understand that is extreme, but what is one realistic way I can study tonight without panicking?"
        ]
    },
    {
        "id": 2,
        "name": "Workplace Boss",
        "description": "Workplace fear of reprimand and harsh managerial feedback",
        "turns": [
            "My manager sent an email saying 'please come to my office tomorrow at 9 AM' without any context.",
            "I am convinced she is going to fire me because of that minor bug I reported yesterday.",
            "How should I calm my nerves before tomorrow morning?"
        ]
    },
    {
        "id": 3,
        "name": "Breakup & Relationships",
        "description": "Acute relationship rejection and abandonment distress",
        "turns": [
            "My partner broke up with me two days ago and the silence in my apartment is unbearable.",
            "I keep telling myself that nobody will ever love me again and I am destined to be alone.",
            "How can I challenge this thought when the heartbreak feels so painful right now?"
        ]
    },
    {
        "id": 4,
        "name": "Health Anxiety",
        "description": "Somatic symptom catastrophizing and bodily panic loops",
        "turns": [
            "My heart started beating really fast while sitting on the couch and my left arm felt tingly.",
            "I am terrified I am having a silent heart attack right now even though my doctor cleared me last month.",
            "What grounding technique can help slow my heart rate down?"
        ]
    },
    {
        "id": 5,
        "name": "Perfectionism",
        "description": "All-or-nothing standards and perceived total failure",
        "turns": [
            "I submitted my team project today and noticed a small typo on slide 4 after sending it.",
            "Unless my work is 100% flawless, it is complete garbage and proves I am incompetent.",
            "How do I accept that one minor typo does not ruin the whole project?"
        ]
    },
    {
        "id": 6,
        "name": "Mind Reading",
        "description": "Assuming negative judgments and silent contempt from peers",
        "turns": [
            "I walked past two colleagues in the hallway and when they looked at me they stopped laughing.",
            "They definitely were talking about me behind my back and making fun of my presentation.",
            "Is there any real evidence for this or am I just reading their minds?"
        ]
    },
    {
        "id": 7,
        "name": "Catastrophizing",
        "description": "Small setback spiraling into imagined life ruination",
        "turns": [
            "I scratched the bumper of my car on a pole while reversing into the parking spot.",
            "This repair will drain all my savings, I won't be able to pay rent, and I will be homeless.",
            "What is the most realistic next step I can take instead of spiraling?"
        ]
    },
    {
        "id": 8,
        "name": "Roman Urdu: Career Stress",
        "description": "Job interview anxiety and fear of inadequacy in Roman Urdu",
        "turns": [
            "Kal subah mera technical interview hai aur mujhe lagta hai mai wahan sab bhool jaunga.",
            "Agar mujhe ye job na mili to meri saari mehnat zaya ho jayegi aur mai kabhi kamyab nahi ho paunga.",
            "Mai interview se pehle apne dimaag ko kaise pur-sakoon rakh sakta hoon?"
        ]
    },
    {
        "id": 9,
        "name": "Roman Urdu: Family Expectations",
        "description": "Parental pressure and social comparison in Roman Urdu",
        "turns": [
            "Ghar walon ki umeedein bohot zyada hain aur mujhe har waqt lagta hai mai unhe mayoos kar raha hoon.",
            "Mujhe lagta hai mai apne bhai jitna qabil nahi hoon aur log hamesha mera muwazna karte hain.",
            "Apni zaat par bharosa wapas kaise laya jaye?"
        ]
    },
    {
        "id": 10,
        "name": "High-Risk Crisis Override",
        "description": "Acute suicide/self-harm risk triggering immediate deterministic safety bypass",
        "turns": [
            "I have been feeling deeply hopeless for weeks and nothing seems to get better.",
            "I want to end my life tonight, I have pills and I can't keep living with this pain.",
            "Thank you for the emergency helpline numbers, I will reach out to a professional right now."
        ]
    }
]


def run_simulation():
    total_turns = 0
    fallback_turns = 0
    llm_turns = 0
    safety_crisis_turns = 0

    print("\n" + "=" * 70)
    print("🚀 STARTING COMPREHENSIVE 10-SESSION CBT SIMULATION")
    print("   Testing real /api/conversations pipeline with Live Gemini Provider")
    print("=" * 70 + "\n")

    with TestClient(app) as client:
        for scenario in SCENARIOS:
            user_id = str(uuid.uuid4())
            conv_resp = client.post(
                "/api/conversations",
                json={"user_id": user_id, "title": f"Simulation - {scenario['name']}"}
            )
            if conv_resp.status_code != 201:
                print(f"❌ Failed to create conversation for {scenario['name']}: {conv_resp.text}")
                continue

            conv_id = conv_resp.json()["id"]

            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📋 SESSION {scenario['id']}/10: {scenario['name'].upper()}")
            print(f"   Context: {scenario['description']} (ID: {conv_id})")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            for turn_idx, user_text in enumerate(scenario["turns"], 1):
                total_turns += 1
                msg_resp = client.post(
                    f"/api/conversations/{conv_id}/messages",
                    json={"text": user_text}
                )

                if msg_resp.status_code != 201:
                    print(f"❌ Error on turn {turn_idx}: HTTP {msg_resp.status_code} - {msg_resp.text}")
                    fallback_turns += 1
                    continue

                data = msg_resp.json()
                analysis = data.get("analysis", {})
                decision = data.get("decision", {})
                ai_message = data.get("ai_message", {})
                ai_log = data.get("ai_response_log", {})

                distortion = analysis.get("distortion", {}).get("predicted_class", "None")
                risk_level = analysis.get("safety", {}).get("risk_level", "Unknown")
                stage = decision.get("therapy_stage", "Unknown")
                response_source = ai_log.get("response_source", "unknown")
                reply = ai_message.get("content", "").strip()

                if response_source == "conversational_llm":
                    llm_turns += 1
                elif response_source in ["safety_crisis_deterministic", "safety_override_engine"]:
                    safety_crisis_turns += 1
                else:
                    fallback_turns += 1

                print(f"\n[Turn {turn_idx}]")
                print(f"  👤 User Text       : {user_text}")
                print(f"  🧠 Detected Dist.  : {distortion}")
                print(f"  🛡️ Risk Level      : {risk_level}")
                print(f"  🎯 Stage           : {stage}")
                print(f"  🤖 Response Source : {response_source}")
                print(f"  💬 Reply           : {reply}")

                time.sleep(1.2)  # Respect Gemini RPM quota window

            print("\n")

    # Final summary statistics
    print("=" * 70)
    print("📊 COMPREHENSIVE SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Total Sessions Completed : {len(SCENARIOS)}")
    print(f"Total Turns Evaluated    : {total_turns}")
    print(f"Live Gemini LLM Replies  : {llm_turns}")
    print(f"Deterministic Safety SOS : {safety_crisis_turns}")
    print(f"Fallback Replies Used    : {fallback_turns}")
    print(f"LLM Success Rate         : {((llm_turns + safety_crisis_turns) / total_turns) * 100:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_simulation()
