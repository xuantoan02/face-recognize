import argparse
import os

import cv2
import numpy as np
import onnx

from src.recognition import ONNXRecognizer


def test_model_structure(model_path):
    print("=== CHECK ONNX FILE ===")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("ONNX model hợp lệ.\n")


def _get_providers(device: str) -> list[str]:
    if device == "gpu":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def test_inference(model_path, image_path=None, providers=None):
    recognizer = ONNXRecognizer(model_path=model_path, providers=providers)

    # Print I/O info
    print("=== INPUTS ===")
    for i, inp in enumerate(recognizer.session.get_inputs()):
        print(f"[{i}] name={inp.name}, shape={inp.shape}, type={inp.type}")

    print("\n=== OUTPUTS ===")
    for i, out in enumerate(recognizer.session.get_outputs()):
        print(f"[{i}] name={out.name}, shape={out.shape}, type={out.type}")

    # Prepare test image
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        print(f"\nDùng ảnh test: {image_path}")
    else:
        img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        print("\nKhông có ảnh test, dùng random input.")

    # Get embedding via ONNXRecognizer
    embedding = recognizer.get_embedding(img)

    print("\n=== EMBEDDING INFO ===")
    print("shape          :", embedding.shape)
    print("L2 norm        :", np.linalg.norm(embedding))
    print("embedding_dim  :", recognizer.embedding_dim)

    print("\nEmbedding sample (10 phần tử đầu):")
    print(embedding[:10])

    # Test batch
    if image_path:
        embeddings = recognizer.get_embeddings_batch([img, img])
        sim = np.dot(embeddings[0], embeddings[1])
        print(f"\n=== BATCH TEST ===")
        print(f"Batch shape    : {embeddings.shape}")
        print(f"Self-similarity: {sim:.6f} (expected ~1.0)")


def main():
    parser = argparse.ArgumentParser(description="Test ONNXRecognizer class")
    parser.add_argument("--model", required=True, help="Đường dẫn file .onnx")
    parser.add_argument("--image", default=None, help="Ảnh test (BGR)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="Thiết bị inference: cpu hoặc gpu (CUDA)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Không thấy model: {args.model}")

    providers = _get_providers(args.device)
    print(f"Device: {args.device.upper()} | Providers: {providers}")

    test_model_structure(args.model)
    test_inference(args.model, args.image, providers=providers)


if __name__ == "__main__":
    main()