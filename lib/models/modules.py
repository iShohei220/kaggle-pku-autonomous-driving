import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, ws=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.ws = ws

    def forward(self, input):
        weight = self.weight
        if self.ws:
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                      keepdim=True).mean(dim=3, keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
            weight = weight / std.expand_as(weight)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class NonLocal2d(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(NonLocal2d, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.key_conv = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.reset_parameters()

    def forward(self, x):
        query = self.query_conv(x).flatten(2)
        key = self.key_conv(x).flatten(2)
        value = self.value_conv(x).flatten(2)

        energy = query.transpose(1, 2).bmm(key)
        weight = F.softmax(energy, dim=-1)
        output = x + value.bmm(weight.transpose(1, 2)).reshape(x.size())

        return output

    def reset_parameters(self):
        nn.init.zeros_(self.value_conv.weight)
        nn.init.zeros_(self.value_conv.bias) 
