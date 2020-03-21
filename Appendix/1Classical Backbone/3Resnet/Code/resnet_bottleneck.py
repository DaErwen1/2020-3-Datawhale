import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(Bottleneck, self).__init__()
        # 网路堆叠层是由1×1、 3×3、 1×1这3个卷积组成的， 中间包含BN层
        self.bottleneck = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, in_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
            )
        self.relu = nn.ReLU(inplace=True)
        # Downsample部分是由一个包含BN层的1×1卷积组成
        self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, 1),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        # 将identity（恒等映射） 与网络堆叠层输出进行相加， 并经过ReLU后输出
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

bottleneck_1_1=Bottleneck(64,256)
input=torch.randn(1,64,56,56)
output=bottleneck_1_1(input)
