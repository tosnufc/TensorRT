import numpy as np
import torch
import torchvision
from torchvision import datasets

# Imagenette folder name -> ImageNet class number
WNID_TO_IMAGENET = {
	"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
        "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
        "n03445777": 574, "n03888257": 701,
}

# Same preprocessing as the 04 file. 
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
dataset = datasets.ImageFolder("/lab/data/imagenette2-320/val", transform=weights.transforms())

# All 3925 images in one batch
loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
images, labels = next(iter(loader))

# Folder index -> ImageNet class number
imagenet_labels = np.array([WNID_TO_IMAGENET[dataset.classes[l]] for l in labels])

np.save("/lab/models/val_images.npy", images.numpy())
np.save("/lab/models/val_labels.npy", imagenet_labels)
print("Saved", images.shape, imagenet_labels.shape)
