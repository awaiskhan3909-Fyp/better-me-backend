"""
==============================================================================
BETTER ME AI: AUTOMATED GOOGLE COLAB MULTI-MODEL BACKEND LAUNCHER
==============================================================================
Runs the full clinical multi-model stack on Google Colab (Tesla T4 GPU):
[1] 4-bit LLaMA-3-8B-CBT-LoRA (CUDA GPU)
[2] BERT Cognitive Distortion Classifier (v2)
[3] BERT Safety Risk & Crisis Classifier (v2)
[4] Contextual Emotional NER (v2)
[5] Cloudflare Public HTTPS Tunnel
==============================================================================
"""

import sys
from run_kaggle import *

if __name__ == "__main__":
    print("""
================================================================================
   🧠 BETTER ME AI — GOOGLE COLAB MULTI-MODEL BACKEND LAUNCHER
================================================================================
""")

    # Clean up lingering processes
    run_command("pkill -9 -f uvicorn")
    run_command("fuser -k 8000/tcp > /dev/null 2>&1")
    time.sleep(1)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    except Exception:
        pass

    check_gpu()
    ensure_llm_endpoint()
    setup_kaggle_models()
    preload_all_models()
    setup_cloudflared()
    tunnel_url = start_tunnel()

    print("━" * 80)
    print("🚀 STEP 6/6: LAUNCHING UVICORN SERVER (LISTENING FOR FRONTEND CHAT REQUESTS)")
    print("━" * 80)
    print("Listening on 0.0.0.0:8000. Incoming requests and GPU inference tokens will print live below:\n")

    subprocess.run([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info"
    ])
