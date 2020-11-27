import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import Conv3D1x1
from models.spec_audio_net import BasicConv2DNet, ResidualSE2DNet
from models.video_net import ResidualSE3DNet, Resnet18, Resnet18Scratch, Resnet18Full
from torchvision.models import resnet18 as resnet2d

class AVNet_Base(nn.Module):
    def __init__(self, norm='batch_renorm', video_nz=784, audio_nz=256, verbose=False):
        super(AVNet_Base, self).__init__()
        self.video_nz = video_nz
        self.audio_nz = audio_nz

        self.norm = norm
        self.verbose = verbose

        # normalization parameters
        self.vid_mean = torch.nn.Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1), requires_grad=False)
        self.vid_std = torch.nn.Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1,1), requires_grad=False)

        self.videonet = self._construct_videonet()
        self.audionet = self._construct_audionet()
        self.mergenet = self._construct_mergenet()
        self.classifier = self._construct_classifier()

        self.audio_keys = []

    def _construct_videonet(self):
        raise NotImplementedError

    def _construct_audionet(self):
        raise NotImplementedError

    def _construct_mergenet(self):
        mergenet = nn.Sequential(nn.Conv1d(in_channels=self.video_nz+self.audio_nz, out_channels=512, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU(inplace=True),
                                )
        return mergenet

    def _construct_classifier(self):
        classifier = nn.Linear(512, 1)
        return classifier

    def forward(self, vars):
        x_im = vars['video']
        x_s = [vars[k] for k in self.audio_keys]

        if self.verbose:
            print("Inputs")
            print("Video size", x_im.shape)
            print("Audio size", [x_.shape for x_ in x_s])
        
        x_im = self._get_visual_features(x_im)
        x_s = self._get_audio_features(x_s)

        if self.verbose:
            print("Video feature size", x_im.shape)
            print("Audio feature size", x_s.shape)

        x = self._merge_features(x_im, x_s)
        
        if self.verbose:
            print("Merged feature size", x.shape)

        return self._classify_features(x)

    def _get_visual_features(self, x):
        x = (x-self.vid_mean)/self.vid_std
        x = self.videonet(x)
        x = x.permute(0,1,3,4,2).contiguous()    # permute time dimension to end
        x = x.view(x.size(0), -1, x.size(4))     # flatten spatial dimensions
        return x

    def _get_audio_features(self, x):
        raise NotImplementedError

    def _merge_features(self, x_im, x_s):
        if x_im.size(2) != x_s.size(2):
            x_im = F.interpolate(x_im, (x_s.size(2),))
        x = torch.cat([x_im, x_s], dim=1)

        return self.mergenet(x)

    def _classify_features(self, x):
        x = F.avg_pool1d(x, kernel_size=(x.size(2),)).view(x.size(0), x.size(1))
        return self.classifier(x)


### Different audio subnetworks

class AVNet_AudioDualGCC(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_AudioDualGCC, self).__init__(*args, **kwargs)
        self.audio_keys = ['mel_spec_avg', 'gcc_phat']

    def _construct_audionet(self):
        timbrenet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=1),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz//2, kernel_size=(1,8), stride=1, padding=0),
                                )
        spatialnet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=1),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz//2, kernel_size=(1,8), stride=1, padding=0),
                                )
        
        return nn.ModuleDict({'timbrenet': timbrenet, 'spatialnet': spatialnet})

    def _get_audio_features(self, x):
        x_flat, gcc = x
        x_flat = self.audionet['timbrenet'](x_flat)
        gcc = self.audionet['spatialnet'](gcc)
        return torch.cat([x_flat, gcc], dim=1).squeeze(-1)

class AVNet_AudioDualMel(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_AudioDualMel, self).__init__(*args, **kwargs)
        self.audio_keys = ['mel_spec_avg', 'mel_spec_left', 'mel_spec_right']

    def _construct_audionet(self):
        timbrenet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=1),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz//2, kernel_size=(1,8), stride=1, padding=0),
                                )
        spatialnet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=1),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz//2, kernel_size=(1,8), stride=1, padding=0),
                                )
        
        return nn.ModuleDict({'timbrenet': timbrenet, 'spatialnet': spatialnet})

    def _get_audio_features(self, x):
        x_flat, x1, x2 = x
        x_s = x1-x2
        x_flat = self.audionet['timbrenet'](x_flat)
        x_s = self.audionet['spatialnet'](x_s)
        return torch.cat([x_flat, x_s], dim=1).squeeze(-1)

class AVNet_AudioDualGCCMel(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_AudioDualGCCMel, self).__init__(*args, **kwargs)
        self.audio_keys = ['mel_spec_avg', 'mel_spec_left', 'mel_spec_right', 'gcc_phat']

    def _construct_audionet(self):
        timbrenet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=1),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz//2, kernel_size=(1,8), stride=1, padding=0),
                                )
        spatialnet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=2),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz//2, kernel_size=(1,8), stride=1, padding=0),
                                )
        
        return nn.ModuleDict({'timbrenet': timbrenet, 'spatialnet': spatialnet})

    def _get_audio_features(self, x):
        x_flat, x1, x2, gcc = x
        x_s = torch.cat([x1 - x2, gcc], dim=1)
        x_flat = self.audionet['timbrenet'](x_flat)
        x_s = self.audionet['spatialnet'](x_s)
        return torch.cat([x_flat, x_s], dim=1).squeeze(-1)

class AVNet_Audio1Stream(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio1Stream, self).__init__(*args, **kwargs)
        self.audio_keys = ['mel_spec_left', 'mel_spec_right']

    def _construct_audionet(self):
        audionet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=2),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz, kernel_size=(1,8), stride=1, padding=0),
                                )
        return audionet

    def _get_audio_features(self, x):
        x1, x2 = x
        x = torch.cat([x1, x2], dim=1)      # concatenate along channel axis
        x = self.audionet(x).squeeze(-1)
        return x

    def get_audio_embeddings(self, x_s):
        x_s = (x_s, x_s)
        return self._get_audio_features(x_s)

    def get_spatial_embeddings(self, x1, x2):
        x_s = (x1, x2)
        return self._get_audio_features(x_s)

class AVNet_Audio2Stream(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio2Stream, self).__init__(*args, **kwargs)
        self.audio_keys = ['mel_spec_left', 'mel_spec_right']

    def _construct_audionet(self):
        assert(self.audio_nz % 2 == 0)
        audionet = nn.Sequential(ResidualSE2DNet(audio_nz=64, norm=self.norm, n_in=1),
                                 nn.Conv2d(in_channels=512, out_channels=self.audio_nz // 2, kernel_size=(1,8), stride=1, padding=0),
                                )
        return audionet

    def _get_audio_features(self, x):
        x1, x2 = x
        batch_size = x1.size(0)
        x = torch.cat([x1, x2], dim=0)      # concatenate along batch axis
        x = self.audionet(x).squeeze(-1)
        x1, x2 = x[:batch_size, :, :], x[batch_size:, :, :]
        return torch.cat([x1, x2], dim=1)

    def get_audio_embeddings(self, x_s):
        return self.audionet(x_s).squeeze(-1)

    def get_spatial_embeddings(self, x1, x2):
        return self._get_audio_features((x1, x2))

### Different visual subnetworks

class AVNet_MCX(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_MCX, self).__init__(*args, **kwargs)

    def _construct_videonet(self):
        assert(self.video_nz == 784)
        return nn.Sequential(ResidualSE3DNet(), Conv3D1x1(in_channels=512, out_channels=16, norm=self.norm))

class AVNet_Resnet18(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_Resnet18, self).__init__(*args, **kwargs)

    def _construct_videonet(self):
        assert(self.video_nz == 784)
        return nn.Sequential(Resnet18(), Conv3D1x1(in_channels=512, out_channels=16, norm=self.norm))

class AVNet_Resnet18_Scratch(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_Resnet18_Scratch, self).__init__(*args, **kwargs)

    def _construct_videonet(self):
        assert(self.video_nz == 784)
        return nn.Sequential(Resnet18Scratch(), Conv3D1x1(in_channels=512, out_channels=16, norm=self.norm))

class AVNet_Resnet18_Full(AVNet_Base):
    def __init__(self, *args, **kwargs):
        super(AVNet_Resnet18_Full, self).__init__(*args, **kwargs)

    def _construct_videonet(self):
        assert(self.video_nz == 512)
        return Resnet18Full()

### Full networks

class AVNet_Audio1Stream_Resnet18(AVNet_Audio1Stream, AVNet_Resnet18):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio1Stream_Resnet18, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.audionet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList([self.videonet])

class AVNet_Audio1Stream_Resnet18_Scratch(AVNet_Audio1Stream, AVNet_Resnet18_Scratch):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio1Stream_Resnet18_Scratch, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.audionet, self.videonet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList()

class AVNet_Audio1Stream_Resnet18_Full(AVNet_Audio1Stream, AVNet_Resnet18_Full):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio1Stream_Resnet18_Full, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.audionet, self.videonet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList()

class AVNet_Audio1Stream_MCX(AVNet_Audio1Stream, AVNet_MCX):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio1Stream_MCX, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.videonet, self.audionet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList()

class AVNet_Audio2Stream_Resnet18(AVNet_Audio2Stream, AVNet_Resnet18):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio2Stream_Resnet18, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.audionet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList([self.videonet])

class AVNet_Audio2Stream_MCX(AVNet_Audio2Stream, AVNet_MCX):
    def __init__(self, *args, **kwargs):
        super(AVNet_Audio2Stream_MCX, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.videonet, self.audionet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList()

class AVNet_AudioDualGCC_Resnet18(AVNet_AudioDualGCC, AVNet_Resnet18):
    def __init__(self, *args, **kwargs):
        super(AVNet_AudioDualGCC_Resnet18, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.audionet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList([self.videonet])

class AVNet_AudioDualMel_Resnet18(AVNet_AudioDualMel, AVNet_Resnet18):
    def __init__(self, *args, **kwargs):
        super(AVNet_AudioDualMel_Resnet18, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.audionet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList([self.videonet])

class AVNet_AudioDualGCCMel_Resnet18(AVNet_AudioDualGCCMel, AVNet_Resnet18):
    def __init__(self, *args, **kwargs):
        super(AVNet_AudioDualGCCMel_Resnet18, self).__init__(*args, **kwargs)
        self.main_net = nn.ModuleList([self.audionet, self.mergenet, self.classifier])
        self.finetune_net = nn.ModuleList([self.videonet])

### Other models

class AVNet_Resnet2D_Two_Stream_Audio(nn.Module):
    def __init__(self, norm='batch_renorm', video_nz=64, audio_nz=64, im_size=(112,112), verbose=False, **kwargs):
        super(AVNet_Resnet2D_Two_Stream_Audio, self).__init__()

        # network parameters
        self.video_nz = video_nz
        self.audio_nz = audio_nz
        self.im_size = im_size
        self.norm = norm

        # normalization parameters
        self.vid_mean = torch.nn.Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1), requires_grad=False)
        self.vid_std = torch.nn.Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1,1), requires_grad=False)

        self.verbose = verbose

        self.videonet = self._construct_videonet()
        self.audionet, self.audionet_avg = self._construct_audionet()

        self.video_reduction = Conv3D1x1(in_channels=video_nz*8, out_channels=32, norm=norm)
        self.mergenet = self._construct_mergenet()
        self.classifier = self._construct_classifier()

        self.main_net = nn.ModuleList(modules=[self.audionet, self.video_reduction, self.mergenet, self.classifier])

    def _construct_videonet(self):
        resnet18 = resnet2d(pretrained=True)

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

    def _construct_audionet(self):
        audionet = nn.Sequential(ResidualSE2DNet(audio_nz=self.audio_nz, norm=self.norm, n_in=2),
                                 nn.Conv2d(in_channels=self.audio_nz*8, out_channels=self.audio_nz*4, kernel_size=(1,8), stride=1, padding=0),
                                )
        audionet_avg = nn.Sequential(ResidualSE2DNet(audio_nz=self.audio_nz, norm=self.norm, n_in=1),
                                 nn.Conv2d(in_channels=self.audio_nz*8, out_channels=self.audio_nz*4, kernel_size=(1,8), stride=1, padding=0),
                                )

        return audionet, audionet_avg

    def _merge_features(self, x_im, x_s, x_s_flat):
        x_im = x_im.view(x_s.size(0), -1, x_im.size(1), x_im.size(2), x_im.size(3))
        x_im = x_im.permute(0,2,1,3,4).contiguous()                       
        x_im = self.video_reduction(x_im) # reduce number of channels
        x_im = x_im.permute(0,1,3,4,2).contiguous()          # permute time dimension to end
        x_im = x_im.view(x_im.size(0), -1, x_im.size(4))     # flatten spatial dimensions

        x_s = x_s.view(-1, x_s.size(1), x_s.size(2))
        x_s_flat = x_s_flat.view(-1, x_s_flat.size(1), x_s_flat.size(2))

        if x_im.size(2) != x_s.size(2):
            x_im = F.interpolate(x_im, (x_s.size(2),), None, mode='linear', align_corners=True)

        x = torch.cat([x_im, x_s, x_s_flat], dim=1)
        return x

    def _construct_mergenet(self):
        mergenet = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU(inplace=True),
                                )
        return mergenet

    def _construct_classifier(self):
        classifier = nn.Linear(512, 1)
        return classifier

    def _normalize_images(self, x):
        x = (x-self.vid_mean)/self.vid_std

        x = F.interpolate(x, size=(x.size(2), self.im_size[0], self.im_size[1]), mode='trilinear')
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        return x

    def _normalize_audio(self, x):
        return x

    def forward(self, x_im, x_s, x_s_flat):
        x_im = self._normalize_images(x_im)
        x_im = self.videonet(x_im)
        
        x_s = self.audionet(x_s)
        x_s_flat = self.audionet_avg(x_s_flat)

        x = self._merge_features(x_im, x_s, x_s_flat)

        x = self.mergenet(x)
        x = F.avg_pool1d(x, kernel_size=(x.size(2),)).view(x.size(0), x.size(1))

        x = self.classifier(x)
        return x

    def get_audio_embeddings(self, x_s):
        x_s = self.audionet_avg(x_s)
        return x_s.squeeze(-1)
        
    def get_spatial_embeddings(self, x1, x2):
        x_s = torch.cat([x1, x2], dim=1)
        x_s = self.audionet(x_s)
        return x_s.squeeze(-1)

model_dict = {'resnet18-1stream': AVNet_Audio1Stream_Resnet18, 'resnet18-2stream': AVNet_Audio2Stream_Resnet18, 'mcx-1stream': AVNet_Audio1Stream_MCX, 'mcx-2stream': AVNet_Audio2Stream_MCX,
              'resnet18-dualgcc': AVNet_AudioDualGCC_Resnet18, 'resnet18-dualmel': AVNet_AudioDualMel_Resnet18, 'resnet18-dualgccmel': AVNet_AudioDualGCCMel_Resnet18,
              'resnet18-double-net': AVNet_Resnet2D_Two_Stream_Audio, 'resnet18-scratch-1stream': AVNet_Audio1Stream_Resnet18_Scratch, 'resnet18-full-1stream': AVNet_Audio1Stream_Resnet18_Full}

def test_model(model_name, dataloader):
    print(model_name)
    net = model_dict[model_name](verbose=True).cuda()
    for sample in dataloader:
        keys = net.audio_keys + ['video']
        vars = {k: sample[k].cuda() for k in keys}
        out = net(vars)
        print(out.shape)
        break

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataloader import ASMR_1M_GulpIO
    
    VIDEODIR = "/trainman-mount/trainman-storage-fb869d29-36ae-41af-a2e0-6a1759be5c83/ASMR_filtered_30K_gulpdir"
    AUDIODIR = "/trainman-mount/trainman-storage-fb869d29-36ae-41af-a2e0-6a1759be5c83/ASMR_filtered_30K_audiodir_16000"
    URLFILE = "/home/splits/ASMR_30K_testset.csv"
    dataset = ASMR_1M_GulpIO(video_dir=VIDEODIR, audio_dir=AUDIODIR, url_file=URLFILE, is_test=False, video_downsample_factor=5, augment_video=True, noisy=True, use_gcc_phat=True, clip_length=2.87)
    dataloader = DataLoader(dataset, 2)

    test_model('resnet18-1stream', dataloader)
    test_model('resnet18-2stream', dataloader)
    test_model('mcx-1stream', dataloader)
    test_model('mcx-2stream', dataloader)
    test_model('resnet18-dualgcc', dataloader)
    test_model('resnet18-dualmel', dataloader)
    test_model('resnet18-dualgccmel', dataloader)
