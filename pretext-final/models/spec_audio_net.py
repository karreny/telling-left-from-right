import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import BasicBlock2D, ResidualSEBlock2D, get_norm_dict

class BasicConv2DNet(nn.Module):
    def __init__(self, audio_nz=64, norm='batch_renorm', n_in=1):
        super(BasicConv2DNet, self).__init__()
        Norm2d = get_norm_dict()[norm][1]

        self.audio_nz = audio_nz
        self.norm = norm

        self.audionet = nn.Sequential(BasicBlock2D(in_channels=n_in, out_channels=self.audio_nz, kernel=(3,3), stride=1, padding=1, norm=self.norm),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      BasicBlock2D(in_channels=self.audio_nz, out_channels=self.audio_nz*2, kernel=(3,3), stride=1, padding=1, norm=self.norm),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      BasicBlock2D(in_channels=self.audio_nz*2, out_channels=self.audio_nz*4, kernel=(3,3), stride=1, padding=1, norm=self.norm),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      BasicBlock2D(in_channels=self.audio_nz*4, out_channels=self.audio_nz*8, kernel=(3,3), stride=1, padding=1, norm=self.norm),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      )

    def forward(self, x):
        return self.audionet(x)

class ResidualSE2DNet(nn.Module):
    def __init__(self, audio_nz=64, norm='batch_renorm', n_in=1):
        super(ResidualSE2DNet, self).__init__()
        Norm2d = get_norm_dict()[norm][1]

        self.audio_nz = audio_nz
        self.norm = norm

        self.audionet = nn.Sequential(ResidualSEBlock2D(in_channels=n_in, out_channels=self.audio_nz, kernel=(3,3), stride=2, padding=1, norm=self.norm),
                                      ResidualSEBlock2D(in_channels=self.audio_nz, out_channels=self.audio_nz*2, kernel=(3,3), stride=2, padding=1, norm=self.norm),
                                      ResidualSEBlock2D(in_channels=self.audio_nz*2, out_channels=self.audio_nz*4, kernel=(3,3), stride=2, padding=1, norm=self.norm),
                                      ResidualSEBlock2D(in_channels=self.audio_nz*4, out_channels=self.audio_nz*8, kernel=(3,3), stride=2, padding=1, norm=self.norm),
                                      )

    def forward(self, x):
        return self.audionet(x)

