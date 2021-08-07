from models import UnetAdaptiveBins
import model_io
from PIL import Image
import torch
import os
from torchvision.utils import save_image
import numpy as np 
from torchvision.datasets import ImageFolder
import torchvision.transforms as transform
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256 

from torchvision.utils import make_grid
unloader = transform.ToPILImage()
loader = transform.Compose([transform.ToTensor()])  

def image_loader(image_name,device):
    size=640, 480
    image = Image.open(image_name).convert('RGB')
    image = image.resize(size, Image.ANTIALIAS)
    image = loader(image).unsqueeze(0)
    
    return image.to(device, torch.float)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

device=get_default_device()


model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "../gdrive/MyDrive/pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)
model.to(device)


save_folder_directories=os.listdir('/content/gdrive/MyDrive/Depth')
divided_directories=os.listdir('../gdrive/MyDrive/dataflow/')



#for divided_directory in divided_directories:
divided_directory='1-200' 
dataset_folder='../gdrive/MyDrive/dataflow/'+divided_directory
sub_directories=os.listdir(dataset_folder)
for directories in sub_directories:
  if not directories in save_folder_directories:

    
    print('Directory '+directories)
    dataset_folder='../gdrive/MyDrive/dataflow/'+divided_directory+"/"+directories+'/imgs/'
    save_folder='/content/gdrive/MyDrive/Depth'+"/"+directories
    if(not os.path.isdir(save_folder)):
        print("no save sub folder ")
        os.mkdir(save_folder)

    for i in range(1,41):
      print("file "+str(i))
      
      
      
      save_file_name=save_folder+"/"+str(i).zfill(3)+'.png'
      file_name=dataset_folder+str(i).zfill(3)+'.jpg'
      image=image_loader(file_name,device)
      _,depth=model(image)
      
      depth=depth.squeeze(0).squeeze(0)
      
      mpimg.imsave(save_file_name,depth.detach().cpu())
      print('saved '+str(i))
  
    
    
  

