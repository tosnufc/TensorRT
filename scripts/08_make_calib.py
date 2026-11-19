import numpy as np
import torch
import torchvision
from torchvision import datasets

# Same preprocessing as the model expects (resize, crop, normalize)
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
dataset = datasets.ImageFolder("/lab/data/imagenette2-320/train", transform=weights.transforms())

# 500 random TRAIN images (fixed seed -> same set ever run)
generator = torch.Generator().manual_seed(0)
loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True, generator=generator)
images, labels = next(iter(loader))

np.save("/lab/models/calib.npy", images.numpy())
print("Saved", images.shape, images.dtype)
print("Images per class:", torch.bincount(labels).tolist())


