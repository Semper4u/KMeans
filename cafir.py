import torchvision
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils import data
import math

cifar10_classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                 512, 512, 512, 'M', 512, 512, 512, 'M']}


# 加载数据
def load_data(path, batch_size, flag):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=False, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=transform)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    if flag == 0:
        return trainloader
    elif flag == 1:
        return testloader
    else:
        raise ValueError("the flag can only be 0 or 1, 0 for train data and 1 for test data")


# 定义VGG16网络结构
class VGG(torch.nn.Module):
    def __init__(self, net_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)     # 将通过卷积激活池化后的图像命名为特征层
        self.classifier = torch.nn.Sequential(     # 定义分类期序列，主要为全连接层
            torch.nn.Dropout(),
            torch.nn.Linear(512, 512),              # full connection1
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 512),             # full connection 2
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 10),              # full connection 3 最后输出为10个类别
        )
        for m in self.modules():                        # 自定义权重， 初始化，只针对卷积层
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)                        # 前向传播时， 先通过卷积激活池化等
        # print(x.size())                             # 进入全连接层前， 需要将批次提取出来，从四维转换为二维
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg['VGG16']:
            if v == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           torch.nn.BatchNorm2d(v),                                         # 批量标准化
                           torch.nn.ReLU(inplace=True)]                                     # inplace为true时，将会改变原数据
                in_channels = v
        return torch.nn.Sequential(*layers)                                                 # 拆包


def training(path, batch_size, train_flag, test_flag, learning_rate, momentum, epochs):

    net = VGG('VGG16')                                                                      # 实例化卷积网络结构类
    critertion = torch.nn.CrossEntropyLoss()                                                # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)      # 定义优化器
    trainloader = load_data(path, batch_size, train_flag)                                   # 加载数据

    for epoch in range(epochs):
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()

            outputs = net(inputs)                                          # forward
            loss = critertion(outputs, labels)                             # compute loss
            loss.backward()                                                # backward
            optimizer.step()                                               # grad optimizer

            train_loss += loss.item()

            if batch_idx % 100 == 99:
                print('[%d %d] loss: %.3f' % (epoch+1, batch_idx+1, train_loss/100))
                train_loss = 0.0

    print('finishing training---------')

    # testing
    correct = 0                                                              # 总的准确率
    total = 0                                                                # 类别总数
    class_correct = list(0. for i in range(10))                              # 每个类别的准确率
    class_total = list(0. for i in range(10))                                # 每个类别的数目

    testloader = load_data(path, batch_size, test_flag)
    with torch.no_grad():
        for test_images, test_labels in testloader:
            test_output = net(test_images)
            class_number, predict = torch.max(test_output, 1)
            total += test_labels.size(0)
            correct += (predict == test_labels).sum().item()
            c = (predict == test_labels).squeeze()                          # 缩维，减去维度为1的维数
            for i in range(batch_size):
                lab = test_labels[i]
                class_correct[lab] += c[i].item()                            # 统计每个类别预测正确的数目
                class_total[lab] += 1

        for i in range(10):
            print('accuracy of %.5s : %.2d%% ' % (cifar10_classes[i], 100*class_correct[i]/class_total[i]))

        print('-'*100)
        print('accuracy of the network on the 10000 test images test by VGG16 is : %d %%' % (100*correct/total))


if __name__ == '__main__':
    training('./data/', 100, 0, 1, 0.001, 0.9, 10)
