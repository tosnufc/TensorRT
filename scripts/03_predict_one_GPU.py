import torch
import torchvision
from PIL import Image

# 1.  Load the model
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=weights).eval()
model = model.cuda()

# 2. Load one image (a golf ball)
img = Image.open("/lab/data/imagenette2-320/val/n03445777/" + "ILSVRC2012_val_00041672.JPEG")

# 3. Proprocess it the way the model expects
x = weights.transforms()(img).unsqueeze(0).cuda()
print("Input shape:", x.shape)

# 4. Predict
logits = model(x)
class_id = logits.argmax().item()
print("Predicted:", class_id, weights.meta["categories"][class_id])
