import torch
from torch import nn
import torch.nn.functional as F
# 构建基础的卷积模块， 与Inception v2的基础模块相比， 增加了BN层
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):#有很多需要设置的参数，不能直接debug，加入input后可以debug
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inceptionv2(nn.Module):
    def __init__(self):
        super(Inceptionv2, self).__init__()
        self.branch1 = BasicConv2d(192, 96, 1, 0) # 对应1x1卷积分支
        # 对应1x1卷积与3x3卷积分支
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 48, 1, 0),
            BasicConv2d(48, 64, 3, 1)
        )
        #对应1x1卷积、3x3卷积与3x3卷积分支
        self.branch3 = nn.Sequential(
            BasicConv2d(192, 64, 1, 0),
            BasicConv2d(64, 96, 3, 1),
            BasicConv2d(96, 96, 3, 1)
        )
        #对应3x3平均池化与1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, 1, 0)
        )
    # 前向过程， 将4个分支进行torch.cat()拼接起来
    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

input=torch.randn(1,192,32,32)
net_inceptionv2=Inceptionv2()#前面定义的一个网络

output=net_inceptionv2(input)#这样能运行，Inceptionv2(input)就不行？
#一定要先实例化这个类，不能直接调用。类不是函数
