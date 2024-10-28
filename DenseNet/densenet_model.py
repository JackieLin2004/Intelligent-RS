import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    # DenseNet的基本构建单元，用于特征提取和通道扩展
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # 第一个批归一化和1x1卷积（瓶颈层），减少通道数
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        # 第二个批归一化和3x3卷积，增加新特征
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        # Dropout率，用于在训练中减少过拟合
        self.drop_rate = drop_rate

    def forward(self, x):
        # 通过瓶颈层（1x1卷积）生成新特征
        new_features = self.conv1(F.relu(self.bn1(x)))
        # 通过3x3卷积生成新特征
        new_features = self.conv2(F.relu(self.bn2(new_features)))
        # 如果设置了dropout，则进行随机丢弃
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        # 将输入与新特征在通道维度上拼接（dense连接）
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Module):
    # DenseNet的密集块（DenseBlock），包含多个DenseLayer
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        # 逐层构建 DenseLayer
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(layer)
        # 使用 nn.Sequential 顺序执行所有层
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    # DenseBlock之间的过渡层，用于缩小特征图尺寸和通道数
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        # 批归一化和1x1卷积，将通道数减少一半
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        # 2x2平均池化层，用于缩小特征图尺寸
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 通过批归一化和1x1卷积，然后缩小尺寸
        x = self.conv(F.relu(self.bn(x)))
        return self.pool(x)


class DenseNet(nn.Module):
    # DenseNet主类，组合DenseBlock和Transition，实现整体结构
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1000):
        super(DenseNet, self).__init__()
        # 初始卷积层，缩小尺寸并增加通道数
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 逐个添加 DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 使用 DenseBlock 增加特征
            block = _DenseBlock(num_layers=num_layers, in_channels=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i + 1}', block)
            # 更新通道数
            num_features = num_features + num_layers * growth_rate
            # 如果不是最后一个块，则添加过渡层
            if i != len(block_config) - 1:
                trans = _Transition(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # 最后的批归一化层
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # 分类层，将特征映射到类别数
        self.classifier = nn.Linear(num_features, num_classes)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 提取特征
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 全局平均池化，将特征缩小到1x1
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # 展平特征并通过全连接层分类
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# DenseNet 变体生成函数
def densenet121(num_classes=1000):
    # DenseNet-121配置，增长率为32，层数配置为(6, 12, 24, 16)
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, num_classes=num_classes)


def densenet169(num_classes=1000):
    # DenseNet-169配置，增长率为32，层数配置为(6, 12, 32, 32)
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, num_classes=num_classes)


def densenet201(num_classes=1000):
    # DenseNet-201配置，增长率为32，层数配置为(6, 12, 48, 32)
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, num_classes=num_classes)


def densenet161(num_classes=1000):
    # DenseNet-161配置，增长率为48，层数配置为(6, 12, 36, 24)，初始特征数为96
    return DenseNet(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, num_classes=num_classes)
