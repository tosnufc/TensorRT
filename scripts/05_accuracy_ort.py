import torch
import torchvision
import onnxruntime as ort
from torchvision import datasets

# Imagenette folder name -> ImageNet class number
WNID_TO_IMAGENET = {
	"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
        "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
        "n03445777": 574, "n03888257": 701,
}

# 1. Model on the GPU
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
session = ort.InferenceSession("/lab/models/resnet50.onnx", providers=["CUDAExecutionProvider"])
print(session.get_providers())

# 2. All validation images, preprocessed, in batches of 32
dataset = datasets.ImageFolder("/lab/data/imagenette2-320/val", transform=weights.transforms())
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# 3. Predict every batch and count correct answers
correct = 0
for images, labels in loader:
	predicted = session.run(["logits"], {"input": images.numpy()})[0].argmax(axis=1)
	for p, l in zip(predicted, labels):
		if p.item() == WNID_TO_IMAGENET[dataset.classes[l]]:
				correct += 1
print(f"Correct: {correct} / {len(dataset)}")
print(f"Accuracy: {100* correct / len(dataset)}%")
