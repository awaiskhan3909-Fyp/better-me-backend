import sys
from pathlib import Path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

def run_test():
    client = TestClient(app)
    conv = client.post("/api/conversations", json={"title": "Wife Text Delay Session"}).json()
    cid = conv["id"]

    steps = [
        "hi",
        "i am feeling ignored",
        "i text my my wife yestarday ,but she replied me back 1 day after.",
        "i feel that she is not interested in me anymore.",
        "i feel like i dissoppinted. or i am feeling that she don't love me",
        "that may be i am not capable of her, or she is losing interest in me",
        "i am still hoping that their would be a chance that she loves me",
        "i should talk with her",
        "yes i can write it down"
    ]

    print("\n" + "=" * 80)
    print("CONVERSATION SIMULATION: WIFE TEXT DELAY SCENARIO")
    print("=" * 80 + "\n")

    for i, msg in enumerate(steps, 1):
        resp = client.post(f"/api/conversations/{cid}/messages", json={"text": msg}).json()
        stage = resp["decision"]["therapy_stage"]
        distortion = resp["analysis"]["distortion"]["predicted_class"]
        ai_reply = resp["ai_message"]["content"]
        
        print(f"[{i}] User: {msg}")
        print(f"    🎯 Stage: {stage} | Distortion: {distortion}")
        print(f"    🤖 Better Me: {ai_reply}\n")

if __name__ == "__main__":
    run_test()

