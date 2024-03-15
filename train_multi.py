import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from PIL import Image
import h5py

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sys
sys.path.append('../..')

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import CXRDataset
from libauc.losses import AUCMLoss 
from libauc.optimizers import PESG

def load_data(cxr_folder, txt_folder, batch_size=4, column='impression', pretrained=False, verbose=False): 
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    if pretrained: 
        input_resolution = 224
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        print("Finished image transforms for pretrained model.")
    else: 
        input_resolution = 320
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])
        print("Finished image transforms for clip model.")
        
    # get cxr_filepaths (.h5 files in a directory and its subdirectories)
    cxr_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(cxr_folder):
        filenames = [f for f in filenames if f.endswith('.h5')]
        cxr_filenames.extend(filenames)
    cxr_filepaths = [cxr_folder+f for f in cxr_filenames]
    
    torch_dsets = []
    for cxr_filepath in cxr_filepaths:
        txt_filename = cxr_filepath.split('/')[-1].replace('.h5', '.csv')
        txt_filepath = txt_folder + txt_filename
        dset = CXRDataset(img_path=cxr_filepath,
                          txt_path=txt_filepath, column=column, transform=transform)
        torch_dsets.append(dset)
    
        if verbose: 
            for i in range(len(dset)):
                sample = dset[i]
                plt.imshow(sample['img'][0])
                plt.show()
                print(i, sample['img'].size(), sample['txt'])
                if i == 3:
                    break
    
    loader_params = {'batch_size':batch_size, 'shuffle': True, 'num_workers': 0}
    data_loaders = [data.DataLoader(dset, **loader_params) for dset in torch_dsets]
    return data_loaders, device