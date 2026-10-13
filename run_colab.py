"""
==============================================================================
BETTER ME AI: AUTOMATED GOOGLE COLAB BACKEND LAUNCHER
==============================================================================
Runs the Better Me FastAPI backend with:
- Tesla T4 GPU 4-bit Unsloth LLaMA-3-8B-CBT-LoRA
- BERT Cognitive Distortion Classifier
- BERT Safety Risk Classifier
- Emotional Context NER
- Cloudflare Public HTTPS Tunnel
==============================================================================
"""

import os
import sys
import time
import subprocess
import re


def run_command(cmd, desc=None):
    if desc:
        print(f"[...] {desc}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU DETECTED: {gpu_name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print("⚠️ WARNING: No GPU detected! Please change Runtime type in Colab to 'T4 GPU'!")
            return False
    except ImportError:
        print("⚠️ PyTorch not yet imported.")
        return False


def setup_cloudflared():
    print("\n🌐 Setting Up Cloudflare Tunnel...")
    if not os.path.exists("/usr/local/bin/cloudflared") and not os.path.exists("/usr/bin/cloudflared"):
        print("[...] Downloading cloudflared binary...")
        run_command("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb")
        run_command("dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1")
        run_command("rm -f cloudflared-linux-amd64.deb")
    print("✅ Cloudflared is ready!")


def start_tunnel():
    print("\n🌍 Starting Cloudflare Public HTTPS Tunnel...")
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

    # Poll tunnel.log for the public trycloudflare.com URL
    tunnel_url = None
    for _ in range(35):
        time.sleep(1)
        if os.path.exists("tunnel.log"):
            with open("tunnel.log", "r") as f:
                content = f.read()
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if match:
                    tunnel_url = match.group(0)
                    break

    print("\n" + "=" * 70)
    if tunnel_url:
        print("🎉 BACKEND IS LIVE!")
        print(f"🔗 PUBLIC URL:   {tunnel_url}")
        print(f"🔗 API ENDPOINT: {tunnel_url}/api")
        print(f"📖 SWAGGER DOCS: {tunnel_url}/docs")
    else:
        print("⚠️ Tunnel URL generation is taking longer. Check tunnel.log for details.")
    print("=" * 70 + "\n")
    return tunnel_url


if __name__ == "__main__":
    check_gpu()

    # Auto-detect Google Drive saved models
    drive_paths = [
        "/content/drive/MyDrive/better_me_llama3_merged",
        "/content/drive/MyDrive/better_me_cbt_lora"
    ]
    for p in drive_paths:
        if os.path.exists(p):
            print(f"📁 [DRIVE DETECTED] Found model in Google Drive: '{p}'")
            os.environ["HF_LLM_REPO"] = p
            break

    setup_cloudflared()
    tunnel_url = start_tunnel()

    print("\n🚀 Launching Better Me FastAPI Backend (Live GPU inference)...")
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
