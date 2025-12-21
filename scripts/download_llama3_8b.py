#!/usr/bin/env python3
"""Download Llama 3.1 8B Instruct GGUF model for AlertSage.

This script downloads the optimized GGUF quantized model for cybersecurity incident triage.

Model: Llama-3.1-8B-Instruct-Q6_K.gguf
- Size: ~6.6GB (Q6_K quantization - higher quality)
- Context: 128k tokens
- License: Llama 3.1 Community License (free for commercial use)

Usage:
    python scripts/download_llama3_8b.py
    python scripts/download_llama3_8b.py --models-dir ./models
"""

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, get_token
except ImportError:
    import sys

    print("\n❌ ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    raise SystemExit(1)

REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"


def download_model(models_dir: Path) -> Path:
    """Download Llama 3.1 8B GGUF model from Hugging Face."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / FILENAME

    if target_path.exists():
        print(f"✅ Model already exists at: {target_path}")
        return target_path

    print(f"📥 Downloading {FILENAME} from {REPO_ID}...")
    print(f"   Size: ~6.6GB (Q6_K quantization - higher quality)")
    print(f"   This may take several minutes...\n")

    try:
        # Download with progress bar
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            cache_dir=models_dir.parent / ".cache",
        )

        cached_path = Path(cached_path)

        # Copy to models directory if not already there
        if cached_path.resolve() != target_path.resolve():
            print(f"📦 Copying model to {target_path}...")
            import shutil

            shutil.copy2(cached_path, target_path)

        print("\n✅ Download complete!")
        print(f"   Model path: {target_path}")
        print(f"\n💡 Usage:")
        print(f'   export TRIAGE_LLM_MODEL="{target_path}"')
        print(f'   export NLP_TRIAGE_LLM_BACKEND="{target_path}"')

        return target_path

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Ensure ~6GB free disk space")
        print("  3. Try: huggingface-cli login")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Llama 3.1 8B Instruct GGUF model for AlertSage"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to store the model (default: models)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    models_dir = (project_root / args.models_dir).resolve()

    print("🚀 AlertSage - Llama 3.1 8B Download")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Models dir: {models_dir}\n")

    download_model(models_dir)


if __name__ == "__main__":
    main()
