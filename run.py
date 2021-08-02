from models import UnetAdaptiveBins
import model_io
from PIL import Image
import torch

import matplotlib.pyplot as plt
import numpy as np 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256 

# NYU

# KITTI
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "../drive/MyDrive/pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

test_dataset=ImageFolder("../drive/MyDrive/formattedvelocity",transform=ToTensor())

image,label=test_dataset[20]
print(image.size())
bin_edges, predicted_depth = model(image.unsqueeze(0))
print(predicted_depth.size())

plt.imshow(predicted_depth.squeeze(0).squeeze(0).detach(),cmap='gray')
#plt.imshow(image.permute(1,2,0),cmap='brg')
plt.savefig('../drive/MyDrive/Depth/001.png')
plt.show()


