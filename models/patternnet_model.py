import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Input is a latent vector, we will start processing with a dense layer and then reshape it to a small spatial volume
        self.tconv1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)  # No stride or padding, kernel size 4
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)  # Increase to 8x8
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)  # Increase to 16x16
        self.bn3 = nn.BatchNorm2d(128)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)   # Increase to 32x32
        self.bn4 = nn.BatchNorm2d(64)

        self.tconv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)    # Increase to 64x64
        self.bn5 = nn.BatchNorm2d(32)

        self.tconv6 = nn.ConvTranspose2d(32, 3, 4, 4, 0, bias=False)    # Final size 256x256

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.relu(self.bn5(self.tconv5(x)))
        img = torch.tanh(self.tconv6(x))  # Use tanh for output because it's commonly used for images
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)  # Adapted for 3 input channels

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)  # Final convolution to produce a single output score

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, inplace=True)
        output = torch.sigmoid(self.conv5(x))  # Sigmoid output for binary classification
        return output.squeeze()
