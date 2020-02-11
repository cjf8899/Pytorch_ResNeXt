import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    
    def __init__(self, in_planes, planes, cardinality, stride):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*2)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes, planes*2, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes*2)
            )
        elif in_planes <= planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*2, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes*2)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
#         print(out.shape)
        return out
    
    
class ResNeXt(nn.Module):
    
    def __init__(self, Block, num_blocks, cardinality, num_class=100):
        super(ResNeXt, self).__init__()

        self.Conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(Block, 64, 128, num_blocks[0], cardinality, stride=1)
        self.layer2 = self.make_layer(Block, 256, 256, num_blocks[1], cardinality, stride=2)
        self.layer3 = self.make_layer(Block, 512, 512, num_blocks[2], cardinality, stride=2)
#         self.layer4 = self.make_layer(Block, 1024, 1024, num_blocks[3], cardinality, stride=2)
        self.classifier = nn.Linear(1024, num_class)

    def make_layer(self, Block, in_planes, planes, num_blocks, cardinality, stride):
        layers = []
        layers.append(Block(in_planes, planes, cardinality, stride))
        for i in range(int(num_blocks)-1):
            layers.append(Block(planes*2, planes, cardinality, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.Conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#         out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    
    
def ResNeXt_32x4d():
    return ResNeXt(Block, [3,3,3], 32)

def ResNeXt_16x8d():
    return ResNeXt(Block, [3,3,3], 16)

def test():
    net = ResNeXt3463()
    print(net)
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
