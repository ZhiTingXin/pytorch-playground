import torch
import torch.nn as nn

class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),
        )
        self.downsample = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, padding=1),
                                        nn.Conv2d(128, 128, kernel_size=3,padding=1),
                                        nn.Conv2d(128,128,kernel_size=2,stride=2),
                                        nn.BatchNorm2d(128)
                                        )
    def forward(self, x):
        identity = x
        out = self.layer(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),
        )
        self.downsample = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                        nn.Conv2d(256, 256, kernel_size=3,padding=1),
                                        nn.Conv2d(256,256,kernel_size=2,stride=2),
                                        nn.BatchNorm2d(256)
                                        )
    def forward(self, x):
        identity = x
        out = self.layer(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512,affine=False),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),
        )
        self.downsample = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                        nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                        nn.Conv2d(512,512,kernel_size=2,stride=2),
                                        nn.BatchNorm2d(512)
                                        )
    def forward(self, x):
        identity = x
        out = self.layer(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class Block4(nn.Module):
    def __init__(self):
        super(Block4, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024,affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),
        )
        self.downsample = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=0),
                                        nn.Conv2d(1024,1024,kernel_size=2,stride=2),
                                        nn.BatchNorm2d(1024)
                                        )
    def forward(self, x):
        identity = x
        out = self.layer(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out



class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    


def cifar100(n_channel, pretrained=None):
    layers = [Block1(), Block2(), Block3(), Block4()]
    layers =  nn.Sequential(*layers)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        pthfile = r'/home/xzt/.cache/torch/checkpoints/cifar100-3a55a987.pth'
        model.load_state_dict(torch.load(pthfile,map_location = torch.device('cpu')))
    return model

# if __name__ == '__main__':
#     net = cifar100(128)
#
#     dataiter = iter(testloader)  # 创建一个python迭代器，读入的是我们第一步里面就已经加载好的testloader
#     images, labels = dataiter.next()

