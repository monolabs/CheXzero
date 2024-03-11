import os
from pathlib import Path
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import skimage
from PIL import Image
import torch
import pandas as pd
from tqdm.notebook import tqdm
from data_process import img_to_hdf5
import cv2
from typing import List, Union
import h5py

for split in [
    'train',
    # 'validate',
    # 'test'
    ]:
    cxr_paths = pd.read_csv(f'data/mimic-cxr-data/files/{split}.csv')['Path'].tolist()
    cxr_paths = [p.replace('gs://', 'data/') for p in cxr_paths]
    out_filepath = f'data/mimic-cxr-data/h5files/{split}.h5'
    img_to_hdf5(cxr_paths, out_filepath)