import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############################
###### HELPER MODULE #######
############################

class EvalWrapper(nn.Module):
    def __init__(self, module):
        super(EvalWrapper, self).__init__()
        self.module = module
        for param in self.module.parameters():
            param.requires_grad = False

    # overrides original method so that train is always false
    def train(self, mode=True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    def forward(self, x):
        return self.module(x)


############################
### NORMALIZATION LAYERS ###
############################

class BatchNormPlaceholder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BatchNormPlaceholder, self).__init__()

    def forward(self, x):
        return x

class BatchRenorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-6):
        super(BatchRenorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.fill_(1.0)

        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.bias.data.fill_(0)

        self.register_buffer('running_mean', torch.Tensor(num_features))
        self.register_buffer('running_var', torch.Tensor(num_features))
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def _flatten(self, x):
        raise NotImplementedError

    def _expand(self, x):
        raise NotImplementedError

    def forward(self, x):

        # verify number of channels is correct
        if x.size(1) != self.num_features:
            raise AssertionError("Expected %s channels, input has %s" % (self.num_features, x.size(1)))

        # resize input depending on input dimension
        x_flat = self._flatten(x)

        # update running statistics
        if self.training:
            self.running_var = (1-self.momentum)*self.running_var+self.momentum*torch.var(x_flat, dim=1).data
            self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*torch.mean(x_flat, dim=1).data

        # compute output
        mean = self._expand(Variable(self.running_mean))
        std  = self._expand(Variable(torch.sqrt(self.running_var+self.eps)))
        x = (x-mean)/std

        x = x*self._expand(self.weight) + self._expand(self.bias)
        return x
        
class BatchRenorm1d(BatchRenorm):
    def __init__(self, *args, **kwargs):
        super(BatchRenorm1d, self).__init__(*args, **kwargs)

    def _flatten(self, x):
        return x.permute(1,0,2).contiguous().view(self.num_features, -1).contiguous()

    def _expand(self, x):
        return x.view(1, self.num_features, 1)

class BatchRenorm2d(BatchRenorm):
    def __init__(self, *args, **kwargs):
        super(BatchRenorm2d, self).__init__(*args, **kwargs)

    def _flatten(self, x):
        return x.permute(1,0,2,3).contiguous().view(self.num_features, -1).contiguous()

    def _expand(self, x):
        return x.view(1, self.num_features, 1, 1)

class BatchRenorm3d(BatchRenorm):
    def __init__(self, *args, **kwargs):
        super(BatchRenorm3d, self).__init__(*args, **kwargs)    

    def _flatten(self, x):
        return x.permute(1,0,2,3,4).contiguous().view(self.num_features, -1).contiguous()

    def _expand(self, x):
        return x.view(1, self.num_features, 1, 1, 1)

############################
### CONVOLUTIONAL BLOCKS ###
############################

NORM_DICT = {'batch_norm': (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d), 'batch_renorm': (BatchRenorm1d, BatchRenorm2d, BatchRenorm3d),
             'none': (BatchNormPlaceholder, BatchNormPlaceholder, BatchNormPlaceholder)}

def get_norm_dict():
    return NORM_DICT

class Conv1D1x1(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch_norm'):
        super(Conv1D1x1, self).__init__()
        Norm1d = NORM_DICT[norm][0]
        self.net = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                 Norm1d(out_channels),
                                 nn.ReLU(inplace=True),
                                )
    def forward(self, x):
        return self.net(x)

class Conv3D1x1(nn.Module):
    def __init__(self, in_channels, out_channels, norm='batch_norm'):
        super(Conv3D1x1, self).__init__()
        Norm3d = NORM_DICT[norm][2]
        self.net = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                 Norm3d(out_channels),
                                 nn.ReLU(inplace=True),
                                )
    def forward(self, x):
        return self.net(x)


class UpConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, norm='batch_norm'):
        super(UpConvBlock3D, self).__init__()

        Norm3d = NORM_DICT[norm][2]
        self.conv = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                                     Norm3d(out_channels),
                                     nn.ReLU(inplace=True),
                                 )

    def forward(self, x):
        return self.conv(x)

class BasicBlock1D(nn.Module):
    """ Standard double conv block, 1D """
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, norm='batch_norm'):
        super(BasicBlock1D, self).__init__()

        Norm1d = NORM_DICT[norm][0]
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                                     Norm1d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
                                     Norm1d(out_channels),
                                     nn.ReLU(inplace=True),
                                    )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.conv(x)

class BasicBlock2D(nn.Module):
    """ Standard double conv block, 2D """
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, norm='batch_norm'):
        super(BasicBlock2D, self).__init__()

        Norm2d = NORM_DICT[norm][1]
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                                     Norm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
                                     Norm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                    )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.conv(x)

#####################
###RESIDUAL BLOCKS###
#####################

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(1,1,1), stride=1, padding=0, norm='batch_norm', rescale=False):
        super(ResidualBlock3D, self).__init__()

        Norm3d = NORM_DICT[norm][2]

        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=stride),
                                            Norm3d(out_channels),
                                           )
        else:
            self.downsample = None

        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                                     Norm3d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
                                     Norm3d(out_channels),
                                    )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if self.downsample:
            x = self.conv(x) + self.downsample(x)
        else:
            x = self.conv(x) + x
        return F.relu(x)

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(1,1), stride=1, padding=0, norm='batch_norm', rescale=False):
        super(ResidualBlock2D, self).__init__()

        Norm2d = NORM_DICT[norm][1]

        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride),
                                            Norm2d(out_channels),
                                           )
        else:
            self.downsample = None

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                                     Norm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
                                     Norm2d(out_channels),
                                    )


        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if self.downsample:
            x = self.conv(x) + self.downsample(x)
        else:
            x = self.conv(x) + x
        return F.relu(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, norm='batch_norm', rescale=False):
        super(ResidualBlock1D, self).__init__()

        Norm1d = NORM_DICT[norm][0]

        if (stride!= 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                                            Norm1d(out_channels),
                                            )
        else:
            self.downsample = None

        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                                  Norm1d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding),
                                  Norm1d(out_channels),
                                  )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if self.downsample:
            x = self.conv(x) + self.downsample(x)
        else:
            x = self.conv(x) + x
        return F.relu(x)

########################
###RESIDUAL SE BLOCKS###
########################

class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SELayer2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResidualSEBlock1D(ResidualBlock1D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResidualSEBlock1D, self).__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.se = SELayer1D(channel=out_channels)

    def forward(self, x):
        if self.downsample:
            x = self.se(self.conv(x)) + self.downsample(x)
        else:
            x = self.se(self.conv(x)) + x
        return F.relu(x)

class ResidualSEBlock2D(ResidualBlock2D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResidualSEBlock2D, self).__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.se = SELayer2D(channel=out_channels)

    def forward(self, x):
        if self.downsample:
            x = self.se(self.conv(x)) + self.downsample(x)
        else:
            x = self.se(self.conv(x)) + x
        return F.relu(x)

class ResidualSEBlock3D(ResidualBlock3D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResidualSEBlock3D, self).__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.se = SELayer3D(channel=out_channels)

    def forward(self, x):
        if self.downsample:
            x = self.se(self.conv(x)) + self.downsample(x)
        else:
            x = self.se(self.conv(x)) + x
        return F.relu(x)

##############
###UPSAMPLE###
##############

class Upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):  # pytorch bug fix
            self.scale_factor = tuple(float(sf) for sf in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

