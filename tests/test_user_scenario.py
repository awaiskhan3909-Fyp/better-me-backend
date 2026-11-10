if __name__ == "__main__":
    import requests

    cid = requests.post("http://127.0.0.1:8000/api/conversations", json={"title": "Exact User Scenario Test"}).json()["id"]
    print(f"Testing live API on conversation {cid}...\n")

    test_inputs = [
        "I don't know if you could help me",
        "i drink water",
        "what are you saying",
        "have you done solve my problem ?"
    ]

    for msg in test_inputs:
        res = requests.post(f"http://127.0.0.1:8000/api/conversations/{cid}/messages", json={"text": msg}).json()
        stage = res["decision"]["therapy_stage"]
        reason = res["decision"]["reason"]
        dist = res["analysis"]["distortion"]["predicted_class"]
        ai = res["ai_message"]["content"]
        print(f"👤 User: {msg}")
        print(f"🎯 Stage: {stage} ({reason}) | Distortion: {dist}")
        print(f"🤖 AI: {ai}\n")
