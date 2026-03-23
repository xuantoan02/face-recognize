import subprocess
import sys

cmd = [
    sys.executable,
    "-m",
    "paddle2onnx",
    "--model_dir", "./mobileface_v1.0_infer",
    "--model_filename", "inference.pdmodel",
    "--params_filename", "inference.pdiparams",
    "--save_file", "./mobileface.onnx",
    "--opset_version", "11",
    "--enable_onnx_checker", "True",
]

result = subprocess.run(cmd, check=True)
print("Convert xong: ./mobileface.onnx")