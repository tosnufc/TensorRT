import time
import numpy as np
import onnxruntime as ort

# 1. Model on the GPU
session = ort.InferenceSession("/lab/models/resnet50.onnx", providers=["CUDAExecutionProvider"])
print(session.get_providers())

# 2. Time each batch size
for bs in [1, 8, 32]:
	x = np.random.rand(bs, 3, 224, 224).astype(np.float32)
	for _ in range(20):
		session.run(["logits"], {"input": x})
	start = time.perf_counter()
	for _ in range(100):
		session.run(["logits"], {"input": x})
	ms = (time.perf_counter() - start) / 100 * 1000
	print(f"batch {bs:2d}: {ms:6.2f} ms/batch, {bs * 1000 / ms:7.1f} images/s")
