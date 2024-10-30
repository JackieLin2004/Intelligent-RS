from utils import *
from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from my_dataset import SRDataset
from srresnet_model import SRResNet
from tqdm import tqdm
import time

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 2  # 放大比例
ngpu = 1  # GP数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 测试集目录
    data_folder = "./data/"
    # 预训练模型
    srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srresnet_checkpoint, map_location=torch.device('cpu'))
    srresnet_model = SRResNet(large_kernel_size=large_kernel_size,
                              small_kernel_size=small_kernel_size,
                              n_channels=n_channels,
                              n_blocks=n_blocks,
                              scaling_factor=scaling_factor)
    srresnet_model = srresnet_model.to(device)
    srresnet_model.load_state_dict(checkpoint['model'])

    # 多GPU测试
    if torch.cuda.is_available() and ngpu > 1:
        srresnet_model = nn.DataParallel(srresnet_model, device_ids=list(range(ngpu)))

    srresnet_model.eval()
    model = srresnet_model

    # 定制化数据加载器
    test_dataset = SRDataset(data_folder,
                             split='test',
                             crop_size=0,
                             scaling_factor=2,
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                              pin_memory=True)

    # 记录每个样本 PSNR 和 SSIM值
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # 记录测试时间
    start = time.time()

    # 使用tqdm包装数据加载器
    with tqdm(total=len(test_loader), desc="Evaluating...", unit="batch") as pbar:
        with torch.no_grad():
            # 逐批样本进行推理计算
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # 数据移至默认设备
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                # 前向传播.
                sr_imgs = model(lr_imgs)

                # 计算 PSNR 和 SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"PSNR": f"{PSNRs.avg:.6f}", "SSIM": f"{SSIMs.avg:.6f}"})

    # 输出平均PSNR和SSIM
    print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))
    print('平均单张样本用时  {:.3f} 秒'.format((time.time() - start) / len(test_dataset)))


if __name__ == '__main__':
    main()
