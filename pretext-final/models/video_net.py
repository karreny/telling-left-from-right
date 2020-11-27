import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import ResidualSEBlock3D, get_norm_dict
from torchvision.models import resnet18 as resnet2d

class ResidualSE3DNet(nn.Module):
    def __init__(self, video_nz=64, norm='batch_renorm'):
        super(ResidualSE3DNet, self).__init__()
        Norm3d = get_norm_dict()[norm][2]

        self.video_nz = video_nz
        self.norm = norm

        self.videonet = nn.Sequential(nn.Conv3d(3, video_nz, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False),
                                      Norm3d(video_nz),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1),
                                      ResidualSEBlock3D(in_channels=video_nz, out_channels=video_nz, kernel=(3,3,3), stride=1, padding=1, norm=norm),
                                      ResidualSEBlock3D(in_channels=video_nz, out_channels=video_nz, kernel=(3,3,3), stride=1, padding=1, norm=norm),
                                      ResidualSEBlock3D(in_channels=video_nz, out_channels=video_nz*2, kernel=(1,3,3), stride=(1,2,2), padding=(0,1,1), norm=norm),
                                      ResidualSEBlock3D(in_channels=video_nz*2, out_channels=video_nz*2, kernel=(1,3,3), stride=1, padding=(0,1,1), norm=norm),
                                      ResidualSEBlock3D(in_channels=video_nz*2, out_channels=video_nz*4, kernel=(1,3,3), stride=(1,2,2), padding=(0,1,1), norm=norm),
                                      ResidualSEBlock3D(in_channels=video_nz*4, out_channels=video_nz*4, kernel=(1,3,3), stride=1, padding=(0,1,1), norm=norm),
                                      ResidualSEBlock3D(in_channels=video_nz*4, out_channels=video_nz*8, kernel=(1,3,3), stride=(1,2,2), padding=(0,1,1), norm=norm),
                                      ResidualSEBlock3D(in_channels=video_nz*8, out_channels=video_nz*8, kernel=(1,3,3), stride=1, padding=(0,1,1), norm=norm),
                                     )

    def forward(self, x):
        return self.videonet(x)

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet = self._construct_videonet()

    def _construct_videonet(self):
        resnet18 = resnet2d(pretrained=True)
        print("Loaded 2D Resnet18 weights")
        videonet = nn.Sequential(resnet18.conv1,
                                 resnet18.bn1,
                                 resnet18.relu,
                                 resnet18.maxpool,
                                 resnet18.layer1,
                                 resnet18.layer2,
                                 resnet18.layer3,
                                 resnet18.layer4,
        )

        return videonet

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.resnet(x)
        x = x.view(batch_size, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0,2,1,3,4).contiguous()

        return x

class Resnet18Scratch(nn.Module):
    def __init__(self):
        super(Resnet18Scratch, self).__init__()
        self.resnet = self._construct_videonet()

    def _construct_videonet(self):
        resnet18 = resnet2d(pretrained=False)
        print("Loaded 2D Resnet18 - training weights from scratch")
        videonet = nn.Sequential(resnet18.conv1,
                                 resnet18.bn1,
                                 resnet18.relu,
                                 resnet18.maxpool,
                                 resnet18.layer1,
                                 resnet18.layer2,
                                 resnet18.layer3,
                                 resnet18.layer4,
        )

        return videonet

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.resnet(x)
        x = x.view(batch_size, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0,2,1,3,4).contiguous()

        return x

class Resnet18Full(nn.Module):
    def __init__(self):
        super(Resnet18Full, self).__init__()
        self.resnet = self._construct_videonet()

    def _construct_videonet(self):
        resnet18 = resnet2d(pretrained=False)
        print("Loaded 2D Resnet18 - training weights from scratch")
        videonet = nn.Sequential(resnet18.conv1,
                                 resnet18.bn1,
                                 resnet18.relu,
                                 resnet18.maxpool,
                                 resnet18.layer1,
                                 resnet18.layer2,
                                 resnet18.layer3,
                                 resnet18.layer4,
                                 resnet18.avgpool,
        )

        return videonet

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.resnet(x)
        x = x.view(batch_size, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0,2,1,3,4).contiguous()

        return x
