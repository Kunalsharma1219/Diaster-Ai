import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34UNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet34(weights="IMAGENET1K_V1")

        self.encoder0 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu
        )
        self.encoder1 = nn.Sequential(
            base.maxpool,
            base.layer1
        )
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.up4 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up2 = self.up_block(128, 64)
        self.up1 = self.up_block(64, 64)

        # 🔥 Extra upsample to restore 512 resolution
        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4)
        d3 = self.up3(d4)
        d2 = self.up2(d3)
        d1 = self.up1(d2)

        d0 = self.up0(d1)

        out = self.final(d0)

        return out