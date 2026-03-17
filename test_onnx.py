import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/modelnew2_onnx.onnx")

input_name = session.get_inputs()[0].name
img = np.random.randn(1,3,112,112).astype(np.float32)

emb = session.run(None, {input_name: img})
print(emb[0].shape)  # (1,512)