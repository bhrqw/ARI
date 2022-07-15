import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm2d

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class RPS_net(nn.Module):

        def __init__(self, num_class):
            super(RPS_net, self).__init__()
            self.init(num_class)

        def init(self, num_class):
            """Initialize all parameters"""

            self.num_class = num_class
            self.sa = SpatialAttention()
            self.bn = BatchNorm2d(1)
            self.sigmoid = nn.Sigmoid()            

            self.out_dim = 170
    
            self.conv0 = nn.Conv2d(3,21, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn0 = nn.BatchNorm2d(21)

            self.conv1_1 = nn.Sequential(nn.Conv2d(21, 21, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(21),nn.ReLU(), nn.Conv2d(21, 21, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(21))
            self.conv1_2 = nn.Sequential(nn.Conv2d(21, 21, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(21),nn.ReLU(), nn.Conv2d(21, 21, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(21))
            
            self.conv1_x = nn.Sequential(nn.Conv2d(21,42, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(42))
            self.conv2_1 = nn.Sequential(nn.Conv2d(21, 42, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(42),nn.ReLU(), nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(42))
            self.conv2_2 = nn.Sequential(nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(42),nn.ReLU(), nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(42))
            
            self.conv2_x = nn.Sequential(nn.Conv2d(42,85, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(85))
            self.conv3_1 = nn.Sequential(nn.Conv2d(42, 85, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(85),nn.ReLU(), nn.Conv2d(85, 85, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(85))
            self.conv3_2 = nn.Sequential(nn.Conv2d(85, 85, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(85),nn.ReLU(), nn.Conv2d(85, 85, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(85))
            
            self.conv3_x = nn.Sequential(nn.Conv2d(85,170, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(170))
            self.conv4_1 = nn.Sequential(nn.Conv2d(85, 170, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(170),nn.ReLU(), nn.Conv2d(170, 170, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(170))
            self.conv4_2 = nn.Sequential(nn.Conv2d(170, 170, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(170),nn.ReLU(), nn.Conv2d(170, 170, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(170))

            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.prelu = nn.PReLU()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(self.out_dim, self.num_class, bias=False)
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            
        def forward(self, x):

           
            mb = x.size()[0]
            
            
            x = self.conv0(x)
            sa_w = self.sa(x)
            sa_w = self.bn(sa_w)
            sa_w = self.sigmoid(sa_w)
            x = sa_w * x
            x = self.bn0(x)
            x = self.relu(x)

            y = self.conv1_1(x)
            x = y + x
            x = self.relu(x)
            y = self.conv1_2(x)
            x = y + x
            x = self.relu(x)
            
            
            y = self.conv2_1(x)
            x = y + self.conv1_x(x)
            x = self.relu(x)
            y = self.conv2_2(x)
            x = y + x
            x = self.relu(x)
            x = self.pool(x)
            
            
            y = self.conv3_1(x)
            x = y + self.conv2_x(x)
            x = self.relu(x)
            y = self.conv3_2(x)
            x = y + x
            x = self.relu(x)
            x = self.pool(x)
            
            
            y = self.conv4_1(x)
            x = y + self.conv3_x(x)
            x = self.relu(x)
            y = self.conv4_2(x)
            x = y + x
            x = self.relu(x)

            
            x = F.avg_pool2d(x.clamp(min=1e-6).pow(3), (x.size(-2), x.size(-1))).pow(1./3)
            x = x.view(-1, self.out_dim)
            x1 = self.fc(x)
            x2 = self.fc(x)
            
            return x2, x1, sa_w


        
        
class RPS_net_mlp(nn.Module):

        def __init__(self):
            super(RPS_net_mlp, self).__init__()
            self.init()

        def init(self):
            """Initialize all parameters"""
            self.mlp1 = nn.Linear(784, 400)
            self.mlp2 = nn.Linear(400, 400)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(400, 10, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.sa = SpatialAttention()
            self.conv0 = nn.Conv2d(1,3, kernel_size=3, stride=1, padding=1,
                               bias=False)
            self.conv1 = nn.Conv2d(3,1, kernel_size=1, stride=1, padding=0,
                               bias=False)
            self.cuda()

        def forward(self, x):

            y = self.conv0(x)
            sa_w = self.sa(y)
            x = sa_w * x
            # x = self.conv1(x)
            x = x.view(-1, 784)
            y = self.mlp1(x)
            y = F.relu(y)
            
            y = self.mlp2(y)
            y = F.relu(y)

            x1 = self.fc(y)
            x2 = self.fc(y)

            return x2, x1, sa_w
        
        
class RPS_net_18(nn.Module):

        def __init__(self, num_class):
            super(RPS_net_18, self).__init__()
            self.init(num_class)

            
        def init(self, num_class):
            """Initialize all parameters"""

            self.num_class = num_class
            

            self.out_dim = 512
    
            self.conv0 = nn.Sequential(nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(64))
            
            self.conv1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(64))
            self.conv1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(64))
            
            self.conv1_x = nn.Sequential(nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(128))
            self.conv2_1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(128))
            self.conv2_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(128))
            
            self.conv2_x = nn.Sequential(nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(256))
            self.conv3_1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(256))
            self.conv3_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(256))
            
            self.conv3_x = nn.Sequential(nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(512))
            self.conv4_1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(512))
            self.conv4_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(512))

            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.prelu = nn.PReLU()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(self.out_dim, 100, bias=False)
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            
        def forward(self, x):

           
            mb = x.size()[0]
            
            
            x = self.conv0(x)
            
            y = self.conv1_1(x)
            x = y + x
            x = self.relu(x)
            y = self.conv1_2(x)
            x = y + x
            x = self.relu(x)
            
            
            y = self.conv2_1(x)
            x = y + self.conv1_x(x)
            x = self.relu(x)
            y = self.conv2_2(x)
            x = y + x
            x = self.relu(x)
            x = self.pool(x)
            
            
            y = self.conv3_1(x)
            x = y + self.conv2_x(x)
            x = self.relu(x)
            y = self.conv3_2(x)
            x = y + x
            x = self.relu(x)
            x = self.pool(x)
            
            
            y = self.conv4_1(x)
            x = y + self.conv3_x(x)
            x = self.relu(x)
            y = self.conv4_2(x)
            x = y + x
            x = self.relu(x)

            
            x = F.avg_pool2d(x.clamp(min=1e-6).pow(3), (x.size(-2), x.size(-1))).pow(1./3)
            x = x.view(-1, self.out_dim)
            x1 = self.fc(x)
            x2 = self.fc(x)
            
            return x2, x1      
                
        
        
        
        
        
        
        
        
        
        
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv0 = nn.Conv2d(3,3, kernel_size=3, stride=1, padding=1, bias=True)
        self.sa = SpatialAttention()
        self.bn = BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        sa_w = self.sa(x)
        sa_w = self.bn(sa_w)
        sa_w = self.sigmoid(sa_w)
        x = sa_w * x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc(x)
        x2 = self.fc(x)

        return x1, x2, sa_w


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)




    