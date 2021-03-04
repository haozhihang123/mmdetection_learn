import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ipdb import set_trace
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class BottleneckA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckB(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckB, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)

        out += residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        
        self.layer1 = self._make_layer(block[0],  64, layers[0])
        self.layer2 = self._make_layer(block[0], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[0], 256, layers[2], stride=2)
        self.layer4 = self._make_layer_detnet(block[1:], 256, layers[3], stride=1)
        self.layer5 = self._make_layer_detnet(block[1:], 256, layers[4], stride=1)
        # Lateral layers
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer6 = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer_detnet(self, block, planes, blocks, stride=1):
        layers = []
        # BottleneckB
        layers.append(block[1](self.in_planes, planes, stride))
        # BottleneckA
        layers.append(block[0](self.in_planes, planes, stride))
        # BottleneckA
        layers.append(block[0](self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        
        #print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)
        #print(f'c2:{c2.shape}')
        c3 = self.layer2(c2)
        #print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)
        #print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)
        #print(f'c5:{c5.shape}')
        c6 = self.layer5(c5)
        
        # Top-down
        m6 = self.latlayer6(c6)
        m5 = self._upsample_add(m6, self.latlayer5(c5))
        m4 = self._upsample_add(m5, self.latlayer4(c4))
        m3 = self._upsample_add(m4, self.latlayer3(c3))
        m2 = self._upsample_add(m3, self.latlayer2(c2))
        # Smooth
        p2 = self.smooth2(m2)
        p3 = self.smooth3(m3)
        p4 = self.smooth4(m4)
        p5 = self.smooth5(m5)
        p6 = self.smooth6(m6)
        # set_trace()
        return p2, p3, p4, p5,p6

def FPN50_detnet():
    Bottlenecks = [Bottleneck, BottleneckA, BottleneckB]
    return FPN(Bottlenecks, [3,4,6,3,3])      

def test():
    net = FPN50_detnet()
    print(net)
    fms = net(Variable(torch.randn(1,3,224,224)))
    for fm in fms:
        print(fm.size())
test()