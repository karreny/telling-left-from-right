import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import UpConvBlock3D, Upsample, get_norm_dict

class AttentionNet(nn.Module):
    def __init__(self, n_in=512, norm='batch_renorm'):
        super(AttentionNet, self).__init__()
        Norm3d = get_norm_dict()[norm][2]

        self.n_in = n_in
        self.norm = norm
        
        self.attention_net = nn.Sequential(UpConvBlock3D(in_channels=n_in, out_channels=n_in//2, kernel=(1,2,2), stride=(1,1,1), padding=(0,0,0), norm=norm),
                                           Upsample(scale_factor=(1,2,2), mode='trilinear'),
                                           nn.Conv3d(in_channels=n_in//2, out_channels=n_in//4, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
                                           Norm3d(n_in//4),
                                           nn.ReLU(inplace=True),
                                           nn.Conv3d(in_channels=n_in//4, out_channels=1, kernel_size=(1,1,1), stride=1, padding=0),
                                           nn.Tanh()
                                          )

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=(1, x.size(3)))
        x = x.unsqueeze(-1)
        x = self.attention_net(x)
        return x+1

