import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from upmixing_models.unet import AudioNet

import sys
sys.path.append('./pretext-final/')
from model import model_dict as avnet_dict

class UpmixUNetBase(nn.Module):
    def __init__(self, pretrained_videonet=None):
        super(UpmixUNetBase, self).__init__()

        self.pretrained_videonet = pretrained_videonet
        self.videonet = self._construct_videonet()
        self.unet = self._construct_unet()

        self.keys = ['video', 'audio_sum_spec']
        
        # normalization parameters
        self.vid_mean = torch.nn.Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1), requires_grad=False)
        self.vid_std = torch.nn.Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1,1), requires_grad=False)


    def _construct_unet(self):
        unet = AudioNet(64, 2, 2, n_feats=784)
        return unet

    def _construct_videonet(self):
        raise NotImplementedError

    def _get_video_features(self, x):
        x = (x-self.vid_mean)/self.vid_std
        batch_size = x.size(0)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.videonet(x)
        x = x.view(batch_size, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0,2,1,3,4).contiguous() 
        return x

    def forward(self, x):
        input = x['audio_sum_spec']
        x_im = self._get_video_features(x['video'])

        mask = self.unet(input, x_im)
        pred_diff_real = input[:,0,:,:] * mask[:,0,:,:] - input[:,1,:,:] * mask[:,1,:,:]
        pred_diff_img = input[:,0,:,:] * mask[:,1,:,:] + input[:,1,:,:] * mask[:,0,:,:]

        return {'pred': torch.cat((pred_diff_real.unsqueeze(1), pred_diff_img.unsqueeze(1)), 1)}

### Video networks

class UpmixNone(UpmixUNetBase):
    def __init__(self, *args, **kwargs):
        super(UpmixNone, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.unet])
        self.finetune_net = nn.ModuleList()
    
    def _construct_videonet(self):
        return None
    
    def _construct_unet(self):
        print("overriding construct unet")
        unet = AudioNet(64, 2, 2, n_feats=0)
        return unet

    def forward(self, x):
        input = x['audio_sum_spec']

        mask = self.unet(input, None)
        pred_diff_real = input[:,0,:,:] * mask[:,0,:,:] - input[:,1,:,:] * mask[:,1,:,:]
        pred_diff_img = input[:,0,:,:] * mask[:,1,:,:] + input[:,1,:,:] * mask[:,0,:,:]

        return {'pred': torch.cat((pred_diff_real.unsqueeze(1), pred_diff_img.unsqueeze(1)), 1)}

class UpmixResnet18Supervised(UpmixUNetBase):
    def __init__(self, *args, **kwargs):
        super(UpmixResnet18Supervised, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.unet])
        self.finetune_net = nn.ModuleList([self.videonet])
    
    def _construct_videonet(self):
        avnet = avnet_dict['resnet18-1stream']()
        layer = list(avnet.videonet.children())[0]
        return layer.resnet

class UpmixResnet18Scratch(UpmixUNetBase):
    def __init__(self, *args, **kwargs):
        super(UpmixResnet18Scratch, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.unet])
        self.finetune_net = nn.ModuleList([self.videonet])
    
    def _construct_videonet(self):
        avnet = avnet_dict['resnet18-scratch-1stream']()

        if self.pretrained_videonet:
            avnet = nn.DataParallel(avnet)
            checkpoint = torch.load(self.pretrained_videonet)
            avnet.load_state_dict(checkpoint['state_dict'])
            print("loaded pretrained model from", self.pretrained_videonet)
            avnet = avnet.module

        layer = list(avnet.videonet.children())[0]
        return layer.resnet
    
class UpmixResnet18Full(UpmixUNetBase):
    def __init__(self, *args, **kwargs):
        super(UpmixResnet18Full, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.unet])
        self.finetune_net = nn.ModuleList([self.videonet])

    def _construct_videonet(self):
        avnet = avnet_dict['resnet18-full-1stream'](video_nz=512)

        if self.pretrained_videonet:
            avnet = nn.DataParallel(avnet)
            checkpoint = torch.load(self.pretrained_videonet)
            avnet.load_state_dict(checkpoint['state_dict'])
            print("loaded pretrained model from", self.pretrained_videonet)
            avnet = avnet.module

        layers = list(avnet.videonet.resnet.children())[:-1]
        return nn.Sequential(*layers)

model_dict = {'upmix-scratch': UpmixResnet18Scratch, 'upmix-full': UpmixResnet18Full, 'upmix-supervised': UpmixResnet18Supervised, 'upmix-none': UpmixNone}

def test_model(name, dataloader, criterion):
    from torch.autograd import Variable

    print(name)
    net = model_dict[name]() 

    for sample in dataloader:
        # process input data
        keys = net.keys + ['audio_diff_spec']
        vars = {k: Variable(sample[k]) for k in keys}
        batch_size = vars['audio_diff_spec'].size(0)

        net.cuda()
        vars = {k: vars[k].cuda() for k in keys}

        # forward pass
        out = net(vars)
        loss = criterion(vars['audio_diff_spec'], out['pred'])

        print(out['pred'].shape)
        print(loss)

        break

if __name__ == "__main__":
    from dataloader import test_dataset
    from torch.utils.data import DataLoader
    import torch.nn as nn

    dataset = test_dataset()
    dataloader = DataLoader(dataset, 5)
    criterion = nn.L1Loss()
    for name in model_dict.keys():
        test_model(name, dataloader, criterion)
