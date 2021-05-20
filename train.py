# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np

import import_ipynb
import transforms as my_T
import dataset
from dataset import Microscopy_dataset
import utils
from utils import get_transforms, my_collate, get_loaders
import model
from model import UNET



# %%
def train():
    running_loss = 0.0
    model.train()
    
    for i, (input, mask) in enumerate(train_loader):
        #input, mask = input[0].to(device=device)[None,:,:,:], mask[0].to(device=device)[None,:,:,:]
        input = input.to(device=device)
        mask = mask.to(device=device)
        
        input.requires_grad = True
        mask.requires_grad = True
        optimizer.zero_grad()

        output = model(input)
        #output = output.squeeze(1)
        #output = output.squeeze(1)
        #mask = mask.squeeze(1)
        #output = output.type(torch.float32)
        #mask = mask.type(torch.long)
        
        loss = criterion(output, mask)
        loss_item = loss.item()

        running_loss += loss_item

        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, iteration: {i} of {len(train_loader)}, loss: {loss_item}")

    training_loss.append(running_loss / len(train_loader))
    
"""    
fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
fig.add_subplot(1, 3, 2)
plt.imshow(masks[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
fig.add_subplot(1, 3, 3)
plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
plt.show(block=True)
plt.savefig('results/train_output.png')
"""

# %%
def evaluate():
    running_loss_eval = 0.0
    model.eval()
    
    for i, (input, mask, name) in enumerate(val_loader):
        input, mask = input[0].to(device=device)[None,:,:,:], mask[0].to(device=device)[None,:,:,:]

        with torch.no_grad():
            output = model(input)
            # torch.where(outputs > 0.5, torch.ones(1).cuda(),torch.zeros(1).cuda())
            loss = criterion(output, mask)
            loss_item = loss.item()

        running_loss_eval += loss_item

        print(f"Eval: {epoch}, iteration: {i} of {len(val_loader)}, loss: {loss_item}")

    eval_loss.append(running_loss_eval / len(val_loader))

""" 
fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.imshow(inputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
fig.add_subplot(1, 3, 2)
plt.imshow(masks[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
fig.add_subplot(1, 3, 3)
plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
plt.show(block=True)
plt.savefig('results/eval_output.png')
"""

# %%
def plot_losses():
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(training_loss, 'r-', label="training_loss",)
    plt.plot(eval_loss, 'b-', label="validation_loss", )
    plt.title("Training and validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f"unet/{attempt}/plots/training_eval_loss_{attempt}.png")
    plt.close()
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(training_loss, 'r-', label="training_loss", )
    plt.title("Training loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig( f"unet/{attempt}/plots/training_loss_{attempt}.png")
    plt.close()

# %%
if __name__ == "__main__":
    
    TRAIN_IMG_DIR = 'train_img/'
    TRAIN_MASK_DIR = 'train_mask/'
    VAL_IMG_DIR = 'val_img/'
    VAL_MASK_DIR = 'val_mask/'
    
    attempt = 2
    epoch = 10
    batch = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 1
    pin_memory = True
    
    #os.makedirs(os.path.join(image_dir, f"unet/{attempt}/images"), exist_ok=True)
    #os.makedirs(os.path.join(image_dir, f"unet/{attempt}/plots"), exist_ok=True)

    print(f"Running on {device}")
    print(f"This is {attempt}. attempt")

    
    train_loader, val_loader = get_loaders(
        train_dir = TRAIN_IMG_DIR,
        train_maskdir = TRAIN_MASK_DIR,
        val_dir = VAL_IMG_DIR,
        val_maskdir = VAL_MASK_DIR,
        batch_size = batch,
        num_workers = workers,
        pin_memory = pin_memory,
    )
    
    
    """
    dataset = Microscopy_dataset(image_dir, mask_dir)

    train_loader = DataLoader(trainset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)
    eval_loader = DataLoader(evalset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)
    """
    
    model = UNET(n_channels=1, n_classes=1).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    #criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    
    training_loss = []
    eval_loss = []

    for epoch in range(epoch):
        train()
        evaluate()
        plot_losses()
        if (epoch % 10) == 0:
            torch.save(model.state_dict(), f"unet_{attempt}_{epoch}.pt")
        else:
            torch.save(model.state_dict(), f"unet_{attempt}.pt")
    print("Done!")

# %%


# %%
