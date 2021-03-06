import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
import timm

from .modules import Conv2d
from .DCNv2.dcn_v2 import DCN


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True


class ResNetFPN(nn.Module):
    def __init__(self, backbone, heads, head_conv=128,
                 num_filters=[256, 256, 256], pretrained=True,
                 dcn=False, gn=False, ws=False, freeze_bn=False):
        super().__init__()

        self.heads = heads

        if backbone == 'resnet18':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif backbone == 'resnet34':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif backbone == 'resnet50':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet101':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet152':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'se_resnext50_32x4d':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'se_resnext101_32x4d':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet34_v1b':
            self.backbone = timm.create_model('gluon_resnet34_v1b', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 512
        elif backbone == 'resnet50_v1d':
            self.backbone = timm.create_model('gluon_resnet50_v1d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet101_v1d':
            self.backbone = timm.create_model('gluon_resnet101_v1d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnext50_32x4d':
            self.backbone = timm.create_model('resnext50_32x4d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnext50d_32x4d':
            self.backbone = timm.create_model('resnext50d_32x4d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'seresnext26_32x4d':
            self.backbone = timm.create_model('seresnext26_32x4d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet18_ctdet':
            self.backbone = models.resnet18()
            state_dict = torch.load('pretrained_weights/ctdet_coco_resdcn18.pth')['state_dict']
            self.backbone.load_state_dict(state_dict, strict=False)
            num_bottleneck_filters = 512
        elif backbone == 'resnet50_maskrcnn':
            self.backbone = models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained).backbone.body
            print(self.backbone)
            num_bottleneck_filters = 2048
        else:
            raise NotImplementedError

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        self.lateral4 = nn.Sequential(
            Conv2d(num_bottleneck_filters, num_filters[0],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters) if gn else nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True))
        self.lateral3 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 2, num_filters[0],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[0]) if gn else nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True))
        self.lateral2 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 4, num_filters[1],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[1]) if gn else nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True))
        self.lateral1 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 8, num_filters[2],
                   kernel_size=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))

        self.decode3 = nn.Sequential(
            DCN(num_filters[0], num_filters[1],
                kernel_size=3, padding=1, stride=1) if dcn else \
            Conv2d(num_filters[0], num_filters[1],
                   kernel_size=3, padding=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[1]) if gn else nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True))
        self.decode2 = nn.Sequential(
            Conv2d(num_filters[1], num_filters[2],
                   kernel_size=3, padding=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[2]) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))
        self.decode1 = nn.Sequential(
            Conv2d(num_filters[2], num_filters[2],
                   kernel_size=3, padding=1, bias=False, ws=ws),
            nn.GroupNorm(32, num_filters[2]) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                Conv2d(num_filters[2], head_conv,
                       kernel_size=3, padding=1, bias=False, ws=ws),
                nn.GroupNorm(32, head_conv) if gn else nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                          kernel_size=1))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        module_names = [n for n, _ in self.backbone.named_modules()]
        if 'layer0' in module_names:
            x1 = self.backbone.layer0(x)
        else:
            x1 = self.backbone.conv1(x)
            x1 = self.backbone.bn1(x1)
            x1 = self.backbone.relu(x1)
            x1 = self.backbone.maxpool(x1)

        x1 = self.backbone.layer1(x1)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        lat4 = self.lateral4(x4)
        lat3 = self.lateral3(x3)
        lat2 = self.lateral2(x2)
        lat1 = self.lateral1(x1)

        map4 = lat4
        map3 = lat3 + F.interpolate(map4, scale_factor=2, mode="nearest")
        map3 = self.decode3(map3)
        map2 = lat2 + F.interpolate(map3, scale_factor=2, mode="nearest")
        map2 = self.decode2(map2)
        map1 = lat1 + F.interpolate(map2, scale_factor=2, mode="nearest")
        map1 = self.decode1(map1)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(map1)
        return ret
