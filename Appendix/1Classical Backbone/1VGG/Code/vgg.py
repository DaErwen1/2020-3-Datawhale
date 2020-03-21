import torch
from torch import nn
class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3#首个卷积层输入维度为3
        out_dim = 64#首个卷积层输出维度64
        # 循环构造卷积层， 一共有13个卷积层
        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            # 在第2、 4、 7、 10、 13个卷积层后增加池化层,可以参考网络架构图，注意i要减一
            if i==1 or i==3 or i==6 or i==9 or i==12:
                layers += [nn.MaxPool2d(2, 2)]#改为尺度的地方都要加MaxPool
                if i!=9:#只有第10层，就是i=9时，不要输出维度*2
                    out_dim*=2
        self.features = nn.Sequential(*layers)#动态参数
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),#224/7=32,经过5次卷积后的MaxPool2d(2,2)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

vgg=VGG(21)
input=torch.randn(1,3,224,224)
scores=vgg(input)
