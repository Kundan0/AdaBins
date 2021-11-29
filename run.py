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
import shutil
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



divided_directories=os.listdir('../gdrive/MyDrive/dataflow/All')
# os.mkdir('/content/gdrive/MyDrive/Depth2')
# os.mkdir('/content/gdrive/MyDrive/Depth2/All')

for divided_directory in divided_directories:
  save_folder_directories=os.listdir('/content/gdrive/MyDrive/Depth2/All')

  if not divided_directory in save_folder_directories:
    print("directory number ",divided_directory)
    save_folder='/content/gdrive/MyDrive/Depth2/All/'+divided_directory
    os.mkdir(save_folder)
    os.mkdir(save_folder+'/imgs')
    dataset_folder='../gdrive/MyDrive/dataflow/All'+divided_directory+'/imgs'
    shutil.copyfile('../gdrive/MyDrive/dataflow/All/'+divided_directory+'/annotations.json','/content/gdrive/MyDrive/Depth2/All/'+divided_directory+'/annotations.json')
    images=os.listdir(dataset_folder)
    for image_name in images:
      

      complete_image_name=dataset_folder+'/'+image_name
      print("dataset image name",complete_image_name)
      #save_file_name=save_folder+"/"+str(i).zfill(3)+'.png'
      save_file_name=save_folder+"/imgs/"+image_name
      print("save image name",save_file_name)
      #file_name=dataset_folder+str(i).zfill(3)+'.jpg'
      image=image_loader(complete_image_name,device)
      _,depth=model(image)
      
      depth=depth.squeeze(0).squeeze(0)
      
      mpimg.imsave(save_file_name,depth.detach().cpu(),cmap='gray')
      print('saved '+image_name)
  
    
    
  

