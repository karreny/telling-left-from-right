import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import ResidualSEBlock3D, Conv3D1x1

class CoordConvNet(nn.Module):
    def __init__(self, n_in, im_size, expand_feats=False, reduce_feats=False, norm='batch_renorm'):
        super(CoordConvNet, self).__init__()
        self.im_size = im_size
        self.expand_feats = expand_feats
        self.reduce_feats = reduce_feats

        x_coor = torch.arange(self.im_size[0]).view(1, 1, 1, -1, 1).float()
        x_coor = 2*(x_coor / (self.im_size[0]-1)) - 1
        x_coor = x_coor.expand(1, 1, 1, self.im_size[0], self.im_size[1])

        y_coor = torch.arange(self.im_size[1]).view(1, 1, 1, 1, -1).float()
        y_coor = 2*(y_coor / (self.im_size[1]-1)) - 1
        y_coor = y_coor.expand(1, 1, 1, self.im_size[0], self.im_size[1])

        self.coor_grid = nn.Parameter(torch.cat([x_coor, y_coor], dim=1), requires_grad=False)
        self.convnet = nn.Sequential(Conv3D1x1(in_channels=n_in+2, out_channels=512, norm=norm),
                                     Conv3D1x1(in_channels=512, out_channels=n_in, norm=norm)
                       )

    def forward(self, x):
        if self.expand_feats:
            x = x.view(x.size(0), x.size(1), x.size(2), 1, 1)
            x = x.expand(x.size(0), x.size(1), x.size(2), self.im_size[0], self.im_size[1])

        coor_grid = self.coor_grid.expand(x.size(0), self.coor_grid.size(1), x.size(2), self.coor_grid.size(3), self.coor_grid.size(4))
        x = torch.cat([x, coor_grid], dim=1)
        x = self.convnet(x)

        if self.reduce_feats:
            x = F.max_pool3d(x, kernel_size=(1, self.im_size[0], self.im_size[1])).view(x.size(0), x.size(1), x.size(2))

        return x

class CoordProdConvNet(nn.Module):
    def __init__(self, n_in, im_size, expand_feats=False, reduce_feats=False, norm='batch_renorm', n_out=None):
        super(CoordProdConvNet, self).__init__()
        self.n_in = n_in
        if not n_out:
            n_out = n_in
        self.n_out = n_out
        self.im_size = im_size
        self.expand_feats = expand_feats
        self.reduce_feats = reduce_feats
        
        ones = torch.ones(1, 1, 1, self.im_size[0], self.im_size[1])
        x_coor = torch.arange(self.im_size[0]).view(1, 1, 1, -1, 1).float()
        x_coor = 2*(x_coor / (self.im_size[0]-1)) - 1
        x_coor = x_coor.expand(1, 1, 1, self.im_size[0], self.im_size[1])
        y_coor = torch.arange(self.im_size[1]).view(1, 1, 1, 1, -1).float()
        y_coor = 2*(y_coor / (self.im_size[1]-1)) - 1
        y_coor = y_coor.expand(1, 1, 1, self.im_size[0], self.im_size[1])
        
        self.coor_grid = nn.Parameter(torch.cat([ones, x_coor, y_coor], dim=1), requires_grad=False)
        self.convnet = nn.Sequential(Conv3D1x1(in_channels=3*n_in+3, out_channels=512, norm=norm),
                                     Conv3D1x1(in_channels=512, out_channels=n_out, norm=norm)
                       )

    def _reshape_and_multiply(self, coor_grid, x):
        coor_grid = coor_grid.unsqueeze(1)
        x = x.unsqueeze(2)
        prod = coor_grid * x
        return prod.view(prod.size(0), x.size(1)*coor_grid.size(2), prod.size(3), prod.size(4), prod.size(5))

    def forward(self, x):
        if self.expand_feats:
            x = x.view(x.size(0), x.size(1), x.size(2), 1, 1)
            x = x.expand(x.size(0), x.size(1), x.size(2), self.im_size[0], self.im_size[1])

        coor_grid = self.coor_grid.expand(x.size(0), self.coor_grid.size(1), x.size(2), self.coor_grid.size(3), self.coor_grid.size(4))
        
        x = self._reshape_and_multiply(coor_grid, x)
        x = torch.cat([x, coor_grid], dim=1)
        x = self.convnet(x)

        if self.reduce_feats:
            x = F.max_pool3d(x, kernel_size=(1, self.im_size[0], self.im_size[1])).view(x.size(0), x.size(1), x.size(2))

        return x

