# %%
import os
import glob
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import draw
from skimage.io import imread
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import import_ipynb
import transforms as my_T
import dataset
from dataset import Microscopy_dataset

# %%
def get_transforms(train=False, rescale_size=(256, 256), yolo=False):
    transforms = []
    if train:
        transforms.append(my_T.Rescale(rescale_size, yolo))
        transforms.append(my_T.Normalize())
        transforms.append(my_T.ToTensor())
    return my_T.Compose(transforms)

# %%
def my_collate(batch):
    image = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return image, targets

# %%
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    num_workers=1,
    pin_memory=True):
    
    train_ds = Microscopy_dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Microscopy_dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


# %%
