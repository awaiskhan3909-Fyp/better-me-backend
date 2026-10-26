"""
==============================================================================
BETTER ME AI: AUTOMATED KAGGLE MULTI-MODEL BACKEND LAUNCHER
==============================================================================
Runs the full clinical multi-model stack on Kaggle (NVIDIA T4 / P100 GPU):
[1] 4-bit LLaMA-3-8B-CBT-LoRA (CUDA GPU)
[2] BERT Cognitive Distortion Classifier (v2)
[3] BERT Safety Risk & Crisis Classifier (v2)
[4] Contextual Emotional NER (v2)
[5] Cloudflare Public HTTPS Tunnel
==============================================================================
"""

import os
import sys
import time
import subprocess
import re
from pathlib import Path


def run_command(cmd, desc=None):
    if desc:
        print(f"[...] {desc}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()


def check_gpu():
    print("━" * 80)
    print("🔍 STEP 1/6: KAGGLE ACCELERATOR & GPU CHECK")
    print("━" * 80)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU DETECTED      : {gpu_name}")
            print(f"✅ TOTAL VRAM        : {vram:.2f} GB")
            print(f"✅ PYTORCH VERSION   : {torch.__version__}")
            print(f"✅ CUDA RUNTIME      : {torch.version.cuda}")
            return True
        else:
            print("⚠️ WARNING: No GPU detected! Please enable GPU in Kaggle:")
            print("👉 Right Sidebar -> Settings -> Accelerator -> Select 'GPU T4 x2' or 'GPU P100'!")
            return False
    except ImportError:
        print("⚠️ PyTorch not yet imported.")
        return False


def setup_kaggle_models():
    print("\n" + "━" * 80)
    print("📁 STEP 2/6: DOWNLOADING & VERIFYING PRE-TRAINED MODELS FROM GOOGLE DRIVE")
    print("━" * 80)

    try:
        import gdown
    except ImportError:
        print("[...] Installing gdown for automated model downloads...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown

    models_base = Path("/kaggle/working/models") if os.path.exists("/kaggle") else Path("./models")
    models_base.mkdir(parents=True, exist_ok=True)

    llama_path = models_base / "better_me_llama3_merged"
    safety_path = models_base / "bert_safety_classifier"
    dist_path = models_base / "bert_cognitive_distortions"

    # 1. LLaMA-3 Merged CBT Model
    if not llama_path.exists() or len(list(llama_path.glob("*"))) == 0:
        print("⏳ [1/3] Downloading LLaMA-3 Merged CBT Model from Google Drive...")
        gdown.download_folder("https://drive.google.com/drive/folders/14RSnQ2XxlxYpdKlN6yejzQUJ_5kMAKmz", output=str(llama_path), quiet=False)
    else:
        print(f"✅ [1/3] LLaMA-3 Merged CBT Model found: '{llama_path}'")

    # 2. BERT Safety Classifier
    if not safety_path.exists() or len(list(safety_path.glob("*"))) == 0:
        print("⏳ [2/3] Downloading BERT Safety Classifier from Google Drive...")
        gdown.download_folder("https://drive.google.com/drive/folders/1Ia5E7XkEdCqONGBY965813Ig8Fyg5hNo", output=str(safety_path), quiet=False)
    else:
        print(f"✅ [2/3] BERT Safety Classifier found: '{safety_path}'")

    # 3. BERT Cognitive Distortions
    if not dist_path.exists() or len(list(dist_path.glob("*"))) == 0:
        print("⏳ [3/3] Downloading BERT Cognitive Distortions from Google Drive...")
        gdown.download_folder("https://drive.google.com/drive/folders/1nyQ121MC-kZYtbDqYHfwUkdLI086WAwV", output=str(dist_path), quiet=False)
    else:
        print(f"✅ [3/3] BERT Cognitive Distortions found: '{dist_path}'")

    os.environ["USE_LOCAL_LLM"] = "true"
    os.environ["LLM_PROVIDER"] = "local_hf"
    os.environ["HF_LLM_REPO"] = str(llama_path)
    os.environ["BERT_SAFETY_PATH"] = str(safety_path)
    os.environ["BERT_DISTORTION_PATH"] = str(dist_path)

    # Symlink to /content/drive/MyDrive/ for universal cross-compatibility
    try:
        colab_compat_dir = Path("/content/drive/MyDrive")
        colab_compat_dir.mkdir(parents=True, exist_ok=True)
        for sub in [llama_path, safety_path, dist_path]:
            link_target = colab_compat_dir / sub.name
            if not link_target.exists():
                os.symlink(str(sub), str(link_target))
    except Exception:
        pass

    print("✅ All 3 models ready in Kaggle working directory!")


def preload_all_models():
    print("\n" + "━" * 80)
    print("⚡ STEP 3/6: LOADING ALL CLINICAL AI MODELS INTO MEMORY")
    print("━" * 80)

    # 1. BERT Cognitive Distortions
    print("\n[1/4] 🧠 Loading BERT Cognitive Distortions Classifier (v2)...")
    try:
        from app.services.distortion_service import distortion_service
        distortion_service.load_model()
        print("      ✅ BERT Cognitive Distortion Classifier Loaded Successfully!")
    except Exception as e:
        print(f"      ⚠️ Distortion classifier load notice: {e}")

    # 2. BERT Safety Risk Classifier
    print("\n[2/4] 🛡️ Loading BERT Safety Risk & Crisis Classifier (v2)...")
    try:
        from app.services.safety_service import safety_service
        safety_service.load_model()
        print("      ✅ BERT Safety Risk Classifier Loaded Successfully!")
    except Exception as e:
        print(f"      ⚠️ Safety classifier load notice: {e}")

    # 3. Contextual Emotional NER
    print("\n[3/4] 🏷️ Loading Contextual Emotional NER Classifier (v2)...")
    try:
        from app.services.ner_service import ner_service
        test_entities = ner_service.extract_entities("I feel anxious and stressed about my exam")
        tags = [f"{e.label}: {e.text}" for e in test_entities]
        print(f"      ✅ NER Engine Verified (Sample Output: {', '.join(tags)})")
    except Exception as e:
        print(f"      ⚠️ NER engine load notice: {e}")

    # 4. LLaMA-3 8B CBT LoRA on CUDA
    print("\n[4/4] 🦥 Loading Fine-Tuned 4-Bit LLaMA-3 CBT Model on GPU...")
    try:
        from app.services.llm_service import llm_service
        if hasattr(llm_service.provider, "_init_local_pipeline"):
            llm_service.provider._init_local_pipeline()

        print("      [...] Pre-warming GPU CUDA kernels with test prompt...")
        test_resp = llm_service.provider.generate(
            prompt="Hello, I feel overwhelmed today.",
            user_text="Hello, I feel overwhelmed today.",
            distortion_name="Emotional Reasoning",
            temperature=0.7,
            max_tokens=25
        )
        print(f"      ✅ Warm-up complete! Test latency: {test_resp.latency_ms:.0f}ms")
        print(f"      ✅ LLaMA-3 4-Bit LoRA Model is 100% READY on GPU!")
    except Exception as e:
        print(f"      ⚠️ LLaMA-3 load note: {e}")


def setup_cloudflared():
    print("\n" + "━" * 80)
    print("🌐 STEP 4/6: SETTING UP CLOUDFLARE PUBLIC HTTPS TUNNEL")
    print("━" * 80)
    if not os.path.exists("/usr/local/bin/cloudflared") and not os.path.exists("/usr/bin/cloudflared"):
        print("[...] Downloading cloudflared Linux binary...")
        run_command("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb")
        run_command("dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1")
        run_command("rm -f cloudflared-linux-amd64.deb")
    print("✅ Cloudflared is installed and ready!")


def start_tunnel():
    print("\n" + "━" * 80)
    print("🌍 STEP 5/6: STARTING PUBLIC HTTPS TUNNEL")
    print("━" * 80)
    run_command("pkill -9 -f cloudflared")
    time.sleep(1)

    if os.path.exists("tunnel.log"):
        try:
            os.remove("tunnel.log")
        except Exception:
            pass

    with open("tunnel.log", "w") as out:
        subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
            stdout=out,
            stderr=subprocess.STDOUT
        )

    tunnel_url = None
    for _ in range(40):
        time.sleep(1)
        if os.path.exists("tunnel.log"):
            with open("tunnel.log", "r") as f:
                content = f.read()
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if match:
                    tunnel_url = match.group(0)
                    break

    print("\n" + "=" * 80)
    if tunnel_url:
        print("🚀 BETTER ME FASTAPI BACKEND IS LIVE!")
        print(f"🔗 PUBLIC URL   : {tunnel_url}")
        print(f"🔗 API ENDPOINT : {tunnel_url}/api")
        print(f"📖 SWAGGER DOCS : {tunnel_url}/docs")
    else:
        print("⚠️ Tunnel URL generation is taking longer. Check tunnel.log for details.")
    print("=" * 80 + "\n")
    return tunnel_url


if __name__ == "__main__":
    print("""
================================================================================
   🧠 BETTER ME AI — KAGGLE MULTI-MODEL BACKEND LAUNCHER
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
    setup_kaggle_models()
    preload_all_models()
    setup_cloudflared()
    tunnel_url = start_tunnel()

    print("━" * 80)
    print("🚀 STEP 6/6: LAUNCHING UVICORN SERVER (LISTENING FOR FRONTEND CHAT REQUESTS)")
    print("━" * 80)
    print("Listening on 0.0.0.0:8000. Incoming requests and GPU inference tokens will print live below:\n")

    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="warning")
