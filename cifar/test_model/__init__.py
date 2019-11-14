from extract_model import cifar100
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

net = cifar100(128,pretrained=True)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR100(root='/home/xzt/cifar100/test',train=False,download=True,transform=transform)

test_size = len(testset)
# need content
print(test_size)


batch_size = 50
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
# give the info of class
# classes = ('plane', 'car', 'bird', 'cat',
# 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

running_corrects = 0.0
for batch_idx, (data, target) in enumerate(testloader):
    indx_target = target.clone()
    outputs = net(Variable(data))
    _,predicted = torch.max(outputs.data, 1)
    running_corrects += torch.sum(predicted == target.data).to(torch.float32)
test_acc = running_corrects/test_size

print('Acc: {:.4f}'.format(
            test_acc))



