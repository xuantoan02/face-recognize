"""
Download ONNX models for the face recognition pipeline.

Downloads:
- BlazeFace (face detection)
- MobileFaceNet (face recognition)
- MiniFASNetV2 (anti-spoofing)
"""

import os
import sys
import urllib.request

# Model URLs — publicly available ONNX models
MODELS = {
    "blazeface_back.onnx": {
        "url": "https://github.com/nicehash/face-recognition/raw/main/models/blazeface_back.onnx",
        "description": "BlazeFace back camera (256x256) — face detection",
        "size_mb": 0.5,
    },
    "mobilefacenet.onnx": {
        "url": "https://github.com/nicehash/face-recognition/raw/main/models/mobilefacenet.onnx",
        "description": "MobileFaceNet — face recognition embeddings",
        "size_mb": 4.5,
    },
    "MiniFASNetV2.onnx": {
        "url": "https://github.com/nicehash/face-recognition/raw/main/models/MiniFASNetV2.onnx",
        "description": "MiniFASNetV2 — anti-spoofing (scale 2.7)",
        "size_mb": 1.5,
    },
    "MiniFASNetV2SE.onnx": {
        "url": "https://github.com/nicehash/face-recognition/raw/main/models/MiniFASNetV2SE.onnx",
        "description": "MiniFASNetV2-SE — anti-spoofing (scale 4.0)",
        "size_mb": 0.6,
    },
    "facial_expression_recognition_mobilefacenet_2022july_int8.onnx": {
        "url": "https://huggingface.co/opencv/facial_expression_recognition/resolve/main/facial_expression_recognition_mobilefacenet_2022july_int8.onnx",
        "description": "Facial Expression Recognition MobileFaceNet INT8 — emotion detection",
        "size_mb": 1.2,
    },
}

# Fallback URLs if primary sources fail
FALLBACK_MODELS = {
    "blazeface_back.onnx": [
        "https://github.com/zineos/blazeface/raw/main/blazeface_back.onnx",
        "https://huggingface.co/nicehash/blazeface/resolve/main/blazeface_back.onnx",
    ],
    "mobilefacenet.onnx": [
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/mobilefacenet-res2-6-10-2-dim512.onnx",
        "https://huggingface.co/nicehash/mobilefacenet/resolve/main/mobilefacenet.onnx",
    ],
    "MiniFASNetV2.onnx": [
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
    ],
    "MiniFASNetV2SE.onnx": [
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV2SE.pth",
    ],
}


def download_file(url: str, dest: str) -> bool:
    """Download a file with progress reporting."""
    try:
        print(f"  Downloading from: {url}")

        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                mb_done = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\r  Progress: {percent:.0f}% ({mb_done:.1f}/{mb_total:.1f} MB)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        return False


def download_models(model_dir: str = "models"):
    """Download all required ONNX models."""
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("Face Recognition Pipeline — Model Downloader")
    print("=" * 60)
    print()

    success_count = 0
    total = len(MODELS)

    for filename, info in MODELS.items():
        dest = os.path.join(model_dir, filename)

        if os.path.exists(dest):
            file_size = os.path.getsize(dest) / (1024 * 1024)
            print(f"✓ {filename} already exists ({file_size:.1f} MB)")
            print(f"  {info['description']}")
            success_count += 1
            continue

        print(f"↓ Downloading {filename} (~{info['size_mb']} MB)")
        print(f"  {info['description']}")

        # Try primary URL
        if download_file(info["url"], dest):
            file_size = os.path.getsize(dest) / (1024 * 1024)
            print(f"  ✓ Saved ({file_size:.1f} MB)")
            success_count += 1
            continue

        # Try fallback URLs
        downloaded = False
        for fallback_url in FALLBACK_MODELS.get(filename, []):
            print(f"  Trying fallback...")
            if download_file(fallback_url, dest):
                file_size = os.path.getsize(dest) / (1024 * 1024)
                print(f"  ✓ Saved ({file_size:.1f} MB)")
                downloaded = True
                success_count += 1
                break

        if not downloaded:
            print(f"  ✗ Failed to download {filename}")
            print(f"    Please download manually from: {info['url']}")

        print()

    print()
    print("=" * 60)
    print(f"Downloaded {success_count}/{total} models to ./{model_dir}/")
    if success_count < total:
        print("⚠  Some models failed. Please download them manually.")
        print("   See README.md for manual download instructions.")
    else:
        print("✓ All models ready!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ONNX models")
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory to save models (default: models)"
    )
    args = parser.parse_args()

    download_models(args.model_dir)
