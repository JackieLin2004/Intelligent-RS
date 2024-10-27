import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    # ResNeXt 的核心瓶颈模块，使用分组卷积实现多路径特征提取
    def __init__(self, in_channels, out_channels, stride=1, groups=32, width_per_group=4):
        super(Bottleneck, self).__init__()
        # 根据 ResNeXt 的设计，通过 groups 和 width_per_group 计算宽度
        width = groups * width_per_group  # 通道数 = 组数 * 每组的通道数

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3 分组卷积，使用 groups 和 width
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = nn.ReLU()(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return nn.ReLU()(out)

class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=32, width_per_group=4):
        super(ResNeXt, self).__init__()
        self.in_channels = 64   # 初始通道数为64

        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差块
        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, groups=groups, width_per_group=width_per_group)
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2, groups=groups, width_per_group=width_per_group)
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=2, groups=groups, width_per_group=width_per_group)
        self.layer4 = self._make_layer(block, 2048, layers[3], stride=2, groups=groups, width_per_group=width_per_group)

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride, groups, width_per_group):
        layers = [block(self.in_channels, out_channels, stride, groups, width_per_group)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, groups=groups, width_per_group=width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ResNeXt-50 和 ResNeXt-101 生成函数
def resnext50_32x4d(num_classes=1000):
    return ResNeXt(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4)

def resnext101_32x8d(num_classes=1000):
    return ResNeXt(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, groups=32, width_per_group=8)
