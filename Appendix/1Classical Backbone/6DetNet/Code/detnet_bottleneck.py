from torch import nn
class DetBottleneck(nn.Module):
    # 初始化时extra为False时为Bottleneck A， 为True时则为Bottleneck B（有1*1的ResNet）
    def __init__(self, inplanes, planes, stride=1, extra=False):
        super(DetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False), 
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, 
                               dilation=2, bias=False),#diation默认=1，无效
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, 1, bias=False),
                nn.BatchNorm2d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.extra = extra
        # Bottleneck B的1×1卷积
        if self.extra:#生成维度相同可以相加的向量
            self.extra_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 对于Bottleneck B来讲， 需要对恒等映射增加卷积处理， 与ResNet类似
        if self.extra:
            identity = self.extra_conv(x)
        else:
            identity = x
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)
        return out

