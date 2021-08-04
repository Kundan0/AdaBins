from models import UnetAdaptiveBins
import model_io
from PIL import Image
import torch

import matplotlib.pyplot as plt
import numpy as np 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utiils.data.dataloader import DataLoader
MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256 

from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
device=get_default_device()

model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "../drive/MyDrive/pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

test_dataset=ImageFolder("../drive/MyDrive/formattedvelocity",transform=ToTensor())
test_dl=DataLoader(test_dataset,4,shuffle=False,num_workers=2,pin_memory=True)
test_dl_cuda=DeviceDataLoader(test_dl,device)
model.to(device)

for images,labels in  test_dl_cuda:
    print(images.size())
    _,depth=model(images)
    break

for i,(images,labels) in  enumerate(test_dl):
    sample_fname,_=test_dl.dataset.samples[i]
    break

print(depth.size())



show_batch(depth)


