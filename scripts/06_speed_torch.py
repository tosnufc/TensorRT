import time
import torch
import torchvision

# 1. Model on the GPU
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=weights).eval().cuda()

# 2. Time each batch size
with torch.no_grad():
	for bs in [1, 8, 32]:
		x = torch.rand(bs, 3, 224, 224, device="cuda")
		for _ in range(20):
			model(x)
		torch.cuda.synchronize()
		start = time.perf_counter()
		for _ in range(100):
			model(x)
		torch.cuda.synchronize()
		ms = (time.perf_counter() - start) / 100 * 1000
		print(f"batch {bs:2d}: {ms:6.2f} ms/batch, {bs * 1000 / ms:7.1f} images/s")
