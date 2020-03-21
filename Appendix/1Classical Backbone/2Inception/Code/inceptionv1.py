import torch
from torch import nn
import torch.nn.functional as F
# 首先定义一个包含conv与ReLU的基础卷积类
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    def forward(self, x):
        x = self.conv(x)#包含卷积和池化两个部分
        return F.relu(x, inplace=True)
# Inceptionv1的类， 初始化时需要提供各个子模块的通道数大小
class Inceptionv1(nn.Module):
     def __init__(self, in_dim, hid_1_1, hid_2_1, hid_2_3, hid_3_1, out_3_5, out_4_1):
          super(Inceptionv1, self).__init__()
          # 下面分别是4个子模块各自的网络定义
          self.branch1x1 = BasicConv2d(in_dim, hid_1_1, 1)
          self.branch3x3 = nn.Sequential(
               BasicConv2d(in_dim, hid_2_1, 1),
               BasicConv2d(hid_2_1, hid_2_3, 3, padding=1)
          )
          self.branch5x5 = nn.Sequential(
               BasicConv2d(in_dim, hid_3_1, 1),
               BasicConv2d(hid_3_1, out_3_5, 5, padding=2)
          )
          self.branch_pool = nn.Sequential(
               nn.MaxPool2d(3, stride=1, padding=1),
               BasicConv2d(in_dim, out_4_1, 1)
          )
     def forward(self, x):
          b1 = self.branch1x1(x)
          b2 = self.branch3x3(x)
          b3 = self.branch5x5(x)
          b4 = self.branch_pool(x)
          # 将这四个子模块沿着通道方向进行拼接
          output = torch.cat((b1, b2, b3, b4), dim=1)
          return output

#if __name__ == '__main__':
net_inceptionv1 = Inceptionv1(3, 64, 32, 64, 64, 96, 32)
input = torch.randn(1, 3, 256, 256)
output = net_inceptionv1(input)

