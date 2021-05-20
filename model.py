# %%
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image
import os
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

# %%
#Test code with generated data for architecture testing

 
"""
# Unet model implementation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
"""
"""
def test_a():
    #x = torch.randn((1, 1, 256, 256))    
    x = torch.randn((3, 1, 256, 256))
    print(x.dtype)
    print(x.size)
    print("dimenze:" + str(x.ndim))
    print(x.shape)
    
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    #assert preds.shape == x.shape 
    
    #yz = preds.detach().numpy()
    #print(yz.shape)

    #arr = yz
    #arr_ = np.squeeze(arr)
    #plt.imshow(arr_)
    #plt.show()
    

#Test code with preprocessed data for architecture testing (from .txt)
def test_b():
    PATH_TO_IMAGES = 'C:\\Users\\Dell\\Desktop\\dataset_zaloha\\val_img\\251.txt'
    #obr = np.array(Image.open(PATH_TO_IMAGES).convert('RGB'))
    obr = np.loadtxt((PATH_TO_IMAGES), dtype=np.float)
    obr = obr.reshape (1, 3, 256, 256).astype('float32')
    obr = torch.from_numpy(obr)
    print("testb dtype::"+str(obr.dtype))
    print("testb size:"+str(obr.size))
    print("testtb ndim:"+str(obr.ndim))
    print("testtb shape:"+str(obr.shape))      

    model = UNET(in_channels=1, out_channels=1)
    preds = model(obr)

    yz = preds.detach().numpy()
    print()
    print(yz.shape)

    arr = yz
    arr_ = np.squeeze(arr)
    plt.imshow(arr_)
    plt.show()

#Test code with preprocessed data for architecture testing (from .tiff)
def test_c(): 
    PATH_TO_IMAGES = 'C:\\Users\\Dell\\Desktop\\dataset_zaloha\\val_img\\val3.tiff'
    obr = np.array(Image.open(PATH_TO_IMAGES).convert("L"))
    #obr = np.loadtxt((PATH_TO_IMAGES), dtype=np.float)
    obr = obr.reshape (1, 1,256, 256).astype('float32')
    obr = torch.from_numpy(obr)
    print("testb dtype::"+str(obr.dtype))
    print("testb size:"+str(obr.size))
    print("testtb ndim:"+str(obr.ndim))
    print("testtb shape:"+str(obr.shape))      

    model = UNET(in_channels=1, out_channels=1)
    preds = model(obr)

    yz = preds.detach().numpy()
    print()
    print(yz.shape)

    arr = yz
    arr_ = np.squeeze(arr)
    plt.imshow(arr_,cmap = "gray")
    plt.show()
    
if __name__ == "__main__":
    #test_a()
    #test_b()
    test_c()
"""