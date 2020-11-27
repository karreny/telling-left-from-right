import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import ResidualBlock1D, ResidualSEBlock1D, BatchRenorm1d, BatchNormPlaceholder, get_norm_dict

class SampleCNN(nn.Module):
    def __init__(self, audio_nz, norm='batch_renorm'):
        super(SampleCNN, self).__init__()
        Norm1d = get_norm_dict()[norm][0]

        self.audio_nz = audio_nz
        self.norm = norm

        self.audionet = nn.Sequential(nn.Conv1d(2, self.audio_nz, kernel_size=3, stride=3, padding=1),
                                  Norm1d(self.audio_nz),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
                                  ResidualBlock1D(in_channels=self.audio_nz, out_channels=self.audio_nz, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz, out_channels=self.audio_nz, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz, out_channels=self.audio_nz*2, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz*2, out_channels=self.audio_nz*2, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz*2, out_channels=self.audio_nz*4, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz*4, out_channels=self.audio_nz*4, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz*4, out_channels=self.audio_nz*8, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz*8, out_channels=self.audio_nz*8, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz*8, out_channels=self.audio_nz*8, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualBlock1D(in_channels=self.audio_nz*8, out_channels=self.audio_nz*8, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  )
    
    def forward(self, x):
        return self.audionet(x)

class SampleCNN_ResidualSE(nn.Module):
    def __init__(self, audio_nz, norm='batch_renorm'):
        super(SampleCNN_ResidualSE, self).__init__()
        Norm1d = get_norm_dict()[norm][0]

        self.audio_nz = audio_nz
        self.norm = norm

        self.audionet = nn.Sequential(nn.Conv1d(2, self.audio_nz, kernel_size=3, stride=3, padding=1),
                                  Norm1d(self.audio_nz),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
                                  ResidualSEBlock1D(in_channels=self.audio_nz, out_channels=self.audio_nz, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz, out_channels=self.audio_nz, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz, out_channels=self.audio_nz*2, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz*2, out_channels=self.audio_nz*2, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz*2, out_channels=self.audio_nz*4, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz*4, out_channels=self.audio_nz*4, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz*4, out_channels=self.audio_nz*8, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz*8, out_channels=self.audio_nz*8, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz*8, out_channels=self.audio_nz*8, kernel=(3,), stride=(3,), padding=1, norm=self.norm),
                                  ResidualSEBlock1D(in_channels=self.audio_nz*8, out_channels=self.audio_nz*8, kernel=(3,), stride=1, padding=1, norm=self.norm),
                                  )

    def forward(self, x):
        return self.audionet(x)
