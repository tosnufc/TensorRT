import torch
import torchvision

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
	model,
	dummy,
	"/lab/models/resnet50.onnx",
	input_names=["input"],
	output_names=["logits"],
	dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
	opset_version=17,
)
print("Exported /lab/models/resnet50.onnx")


