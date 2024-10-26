import torch
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(VGG19, self).__init__()

        # 定义特征提取部分（features），主要是卷积层和池化层
        self.features = nn.Sequential(
            # 第一层卷积，输入为RGB三通道图像[3, H, W]，输出64通道
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化，减少特征图大小一半

            # 第二层卷积，输入64通道，输出128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化

            # 第三层卷积，输入128通道，输出256通道
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化

            # 第四层卷积，输入256通道，输出512通道
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化

            # 第五层卷积，输入512通道，输出512通道
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化
        )

        # 分类部分（classifier），包含全连接层和Dropout层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout层，用于防止过拟合
            nn.Linear(512 * 8 * 8, 4096),  # 全连接层，输入512*8*8（假设输入256x256），输出4096
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout层
            nn.Linear(4096, 4096),  # 全连接层，输出4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 最后一层，全连接到类别数
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 前向传播，依次经过特征提取和分类部分
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平，用于全连接层
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 权重初始化函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果是卷积层，使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 如果是全连接层，使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
