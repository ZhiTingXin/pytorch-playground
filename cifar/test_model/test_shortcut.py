from shortcut_model import cifar100

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    net = cifar100(128)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR100(root='/home/xzt/cifar100/test', train=False, download=True,
                                            transform=transform)

    test_size = len(testset)
    # need content
    print(test_size)

    batch_size = 50
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)
    dataiter = iter(testloader)  # 创建一个python迭代器，读入的是我们第一步里面就已经加载好的testloader
    images, labels = dataiter.next()
    # Make a grid from batch
    out = torchvision.utils.make_grid(images)
    imshow(out,title={x for x in labels})

    outputs = net(Variable(images))  # 注意这里的images是我们从上面获得的那四张图片，所以首先要转化成variable
    _, predicted = torch.max(outputs.data, 1)
    # 这个 _ , predicted是python的一种常用的写法，表示后面的函数其实会返回两个值
    # 但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，我们只关心第二个值predicted
    # 比如 _ ,a = 1,2 这中赋值语句在python中是可以通过的，你只关心后面的等式中的第二个位置的值是多少

    print('Predicted: ', ' '.join('%5s' % predicted[j]for j in range(4)))  # python的字符串格式化