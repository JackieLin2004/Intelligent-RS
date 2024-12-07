import torch
import torch.nn as nn
import torch.optim as optim


class UnetBlock(nn.Module):
    """
    U-Net结构中的一个Block，包括卷积、反卷积、激活函数和批量归一化。
    该Block实现了U-Net的编码器和解码器部分，支持跳跃连接和瓶颈层。
    """

    def __init__(self, filnum, co, submodule=None, in_channel=None, bottleneck=False, outermost=False):
        super().__init__()
        self.outermost = outermost

        if in_channel is None:
            in_channel = filnum

        # 下采样卷积层
        down_conv = nn.Conv2d(in_channel, co, kernel_size=4, stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, True)  # 激活函数：LeakyReLU
        down_norm = nn.BatchNorm2d(co)  # 批量归一化
        up_relu = nn.ReLU(True)  # 上采样时使用ReLU
        up_norm = nn.BatchNorm2d(filnum)

        # 根据不同情况设置模型（最外层、瓶颈层或中间层）
        if outermost:
            up_conv = nn.ConvTranspose2d(co * 2, filnum, kernel_size=4, stride=2, padding=1)
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]  # 最外层输出用Tanh激活
            model = down + [submodule] + up

        elif bottleneck:
            up_conv = nn.ConvTranspose2d(co, filnum, kernel_size=4, stride=2, padding=1, bias=False)
            down = [down_relu, down_conv]  # 瓶颈层只有下采样和上采样
            up = [up_relu, up_conv, up_norm]
            model = down + up

        else:
            up_conv = nn.ConvTranspose2d(co * 2, filnum, kernel_size=4, stride=2, padding=1, bias=False)
            down = [down_relu, down_conv, down_norm]  # 中间层含有批量归一化
            up = [up_relu, up_conv, up_norm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        前向传播：
        如果是最外层，直接返回经过整个Block处理后的输出；
        否则，拼接输入和当前Block的输出。
        """
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)  # 拼接跳跃连接的输出


class Unet(nn.Module):
    """
    U-Net模型，使用多个UnetBlock堆叠而成。
    通过跳跃连接将编码器和解码器部分连接起来。
    """

    def __init__(self, in_channel=1, out_channel=2, layers=8, filnum=64):
        super().__init__()

        # 构建最底层的瓶颈层
        unet_block = UnetBlock(filnum * 8, filnum * 8, bottleneck=True)

        # 构建编码器部分（逐步减小特征图尺寸）
        for i in range(layers - 5):
            unet_block = UnetBlock(filnum * 8, filnum * 8, submodule=unet_block)

        filout = filnum * 8

        # 构建解码器部分（逐步增大特征图尺寸）
        for i in range(3):
            unet_block = UnetBlock(filout // 2, filout, submodule=unet_block)
            filout //= 2

        # 最外层输出
        self.model = UnetBlock(out_channel, filout, in_channel=in_channel, submodule=unet_block, outermost=True)

    def forward(self, x):
        """
        前向传播：
        通过整个U-Net模型进行推理，生成最终的彩色化图像。
        """
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """
    判别器模型，基于PatchGAN。用于判断输入图像是否真实，输入为拼接后的灰度图像和生成的彩色图像。
    """

    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()

        # 定义模型：多层卷积操作，逐步减小特征图尺寸
        model = [self.get_layers(input_c, num_filters, norm=False)]  # 第一层

        # 构建中间的卷积层，逐层加深特征图数量并减小尺寸
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]

        # 最后一层输出一个单通道的预测结果
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        """
        构建卷积层，包含批量归一化和激活函数。
        """
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播：判别器通过多个卷积层判断输入图像是否为真实图像。
        """
        return self.model(x)


class GANLoss(nn.Module):
    """
    GAN的损失函数，使用二元交叉熵损失来计算生成器和判别器的损失。
    """

    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, preds, target_is_real):
        """
        根据目标是否真实，生成对应的标签（真实为1，虚假为0）。
        """
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        """
        计算损失：根据预测结果和真实标签，计算损失。
        """
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


def init_model(net, device, gain=0.02, Gen=False):
    """
    初始化网络权重：使用正态分布初始化卷积层权重，批量归一化层初始化为1。
    """
    net = net.to(device)

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            nn.init.normal_(m.weight.data, mean=0.0, std=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)

    if Gen:
        print(f"Generator initialized with norm initialization")
    else:
        print(f"Discriminator initialized with norm initialization")

    return net


class Colorization_Model(nn.Module):
    """
    图像彩色化模型，结合生成器（U-Net）和判别器（PatchGAN）。
    训练生成器生成彩色化图像，判别器用于判断生成的图像是否真实。
    """

    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        # 初始化生成器
        if net_G is None:
            self.net_G = init_model(Unet(in_channel=1, out_channel=2, layers=8, filnum=64), device=self.device,
                                    Gen=True)
        else:
            self.net_G = net_G.to(self.device)

        # 初始化判别器
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), device=self.device)
        self.GANcriterion = GANLoss().to(self.device)
        self.L1criterion = nn.L1Loss()  # L1损失用于控制彩色化效果
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        """
        设置模型的参数是否需要梯度计算。
        """
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        """
        设置输入数据：灰度图（L）和真实的色度图（ab）。
        """
        self.L = data[0].to(self.device)
        self.ab = data[1].to(self.device)

    def forward(self):
        """
        通过生成器进行前向传播，生成彩色化图像。
        """
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        """
        判别器的反向传播：计算判别器对于真实图像和生成图像的损失。
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)

        # 总损失
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """
        生成器的反向传播：计算生成器的对抗损失和L1损失。
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        """
        训练步骤：
        1. 更新判别器（D）；
        2. 更新生成器（G）。
        """
        self.forward()  # 生成彩色化图像
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()  # 计算并更新判别器损失
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)  # 固定判别器，不计算其梯度
        self.opt_G.zero_grad()
        self.backward_G()  # 计算并更新生成器损失
        self.opt_G.step()
