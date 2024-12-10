import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from models import *
from datasets import *

# 设置日志记录
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 输出到文件
file_handler = logging.FileHandler('training_log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 计算PSNR
def calculate_psnr(img1, img2, data_range=255.0):
    """
    计算 PSNR 指标
    img1: 生成的高分辨率图像
    img2: 真实的高分辨率图像
    data_range: 图像像素范围，通常是 255 或 1，取决于图像的类型
    """
    # 确保图像在 [0, data_range] 范围内
    img1 = img1.detach().cpu().numpy()  # 将tensor转换为NumPy数组
    img2 = img2.detach().cpu().numpy()

    # 对于浮动类型的图像，我们需要乘以 255.0 进行归一化
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0

    # 计算 PSNR
    return psnr(img1, img2, data_range=data_range)
def calculate_ssim(img1, img2):
    """
    计算 SSIM 指标
    img1: 生成的高分辨率图像
    img2: 真实的高分辨率图像
    """
    # 确保图像在 [0, 255] 范围内
    img1 = img1.detach().cpu().numpy()  # 将 tensor 转换为 NumPy 数组
    img2 = img2.detach().cpu().numpy()

    # 对于浮动类型的图像，我们需要乘以 255.0 进行归一化
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0
    
    # 确保图像的尺寸至少为 7x7，或者根据图像大小调整窗口
    min_side = min(img1.shape[1], img1.shape[2])

    # 设置适当的 win_size
    # 如果图像尺寸太小，将 win_size 调整为图像的最小边长，并确保它是奇数
    win_size = min(7, min_side)  # win_size 不超过图像最小边长
    if win_size % 2 == 0:
        win_size -= 1  # 保证 win_size 是奇数

    # 如果图像尺寸小于 7x7，直接设置为图像的最小边长
    if min_side < 7:
        win_size = min_side if min_side % 2 == 1 else min_side - 1

    # 如果图像尺寸更小（例如 1x1 或 2x2），直接跳过 SSIM 计算，返回默认值
    if min_side <= 1:
        return 1.0

    # 计算 SSIM 时明确指定 data_range
    data_range = 255  # 假设像素值范围是 [0, 255]

    # 调用 SSIM 函数
    try:
        return ssim(img1, img2, multichannel=True, win_size=win_size, data_range=data_range)
    except ValueError as e:
        print(f"SSIM error: {e}")
        return 0.0  # 在 SSIM 计算失败时返回一个默认值




if __name__ == '__main__':
    # 省略部分代码
    # 创建保存图片和模型的文件夹
    os.makedirs("images/training", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="resized", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=5000,
                        help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    dataloader = DataLoader(
        ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        ImageDataset("../../data/%s_val" % opt.dataset_name, hr_shape=hr_shape),  # 假设你有一个单独的验证集
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    best_val_loss = -float('inf')  # 初始的最优验证损失为负无穷，避免第一次就保存
    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_loss_content = 0.0
        epoch_loss_GAN = 0.0
        epoch_loss_pixel = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        # 训练阶段
        for i, imgs in enumerate(dataloader):
            batches_done = epoch * len(dataloader) + i

            # 配置模型输入
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------ 训练生成器 ------------------

            optimizer_G.zero_grad()

            # 生成高分辨率图像
            gen_hr = generator(imgs_lr)
            gen_hr = torch.nn.functional.interpolate(gen_hr, size=imgs_hr.shape[2:], mode='bilinear',
                                                     align_corners=False)

            # 计算像素级损失
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < opt.warmup_batches:
                # 热身阶段，只计算像素级损失
                loss_pixel.backward()
                optimizer_G.step()
                logger.info(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [G pixel: {loss_pixel.item()}]")
                continue

            # 从判别器提取有效性预测
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # 对抗损失（相对平均 GAN）
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # 内容损失
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # 生成器的总损失
            loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ------------------ 训练判别器 ------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # 对抗损失（真实图像与伪图像）
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # 判别器的总损失
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # ------------------ 记录训练信息 ------------------

            # 累加每个batch的损失
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_loss_content += loss_content.item()
            epoch_loss_GAN += loss_GAN.item()
            epoch_loss_pixel += loss_pixel.item()

            # 计算PSNR和SSIM
            batch_psnr = calculate_psnr(gen_hr, imgs_hr,data_range=255.0)
            batch_ssim = calculate_ssim(gen_hr, imgs_hr)

            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim

            # logger.info(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
            #              f"[D loss: {loss_D.item()}] [G loss: {loss_G.item()}] "
            #              f"[content: {loss_content.item()}] [adv: {loss_GAN.item()}] [pixel: {loss_pixel.item()}] "
            #              f"[PSNR: {batch_psnr:.2f}] [SSIM: {batch_ssim:.4f}]")

            if batches_done % opt.sample_interval == 0:
                imgs_lr_resized = nn.functional.interpolate(imgs_lr, scale_factor=4)
                # 假设 gen_hr 的目标尺寸是 imgs_hr 的尺寸
                gen_hr_resized = nn.functional.interpolate(gen_hr, size=imgs_lr_resized.shape[2:], mode='bilinear', align_corners=False)

# 然后再进行拼接
                img_grid = denormalize(torch.cat((imgs_lr_resized, gen_hr_resized), -1))

                save_image(img_grid, f"images/training/{batches_done}.png", nrow=1, normalize=False)

        # 计算每个epoch的平均损失和指标
        avg_loss_D = epoch_loss_D / len(dataloader)
        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_content = epoch_loss_content / len(dataloader)
        avg_loss_GAN = epoch_loss_GAN / len(dataloader)
        avg_loss_pixel = epoch_loss_pixel / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)

        logger.info(f"Epoch {epoch}/{opt.n_epochs} - "
                     f"Average D loss: {avg_loss_D:.4f}, Average G loss: {avg_loss_G:.4f}, "
                     f"Average content loss: {avg_loss_content:.4f}, Average GAN loss: {avg_loss_GAN:.4f}, "
                     f"Average pixel loss: {avg_loss_pixel:.4f}, Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

        # 保存最佳模型（依据PSNR/SSIM指标或损失）
        if avg_psnr > best_val_loss:
            best_val_loss = avg_psnr
            logger.info(f"Saving model with improved PSNR: {avg_psnr:.2f}")
            torch.save(generator.state_dict(), f"saved_models/generator_best.pth")
            torch.save(discriminator.state_dict(), f"saved_models/discriminator_best.pth")
