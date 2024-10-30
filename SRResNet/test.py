from utils import *
from srresnet_model import SRResNet
import time
from PIL import Image

# 测试图像
imgPath = './results/test.jpg'

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4  # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 预训练模型
    srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet
    checkpoint = torch.load(srresnet_checkpoint, map_location=torch.device('cpu'))
    srresnet_model = SRResNet(large_kernel_size=large_kernel_size,
                              small_kernel_size=small_kernel_size,
                              n_channels=n_channels,
                              n_blocks=n_blocks,
                              scaling_factor=scaling_factor)
    srresnet_model = srresnet_model.to(device)
    srresnet_model.load_state_dict(checkpoint['model'])

    srresnet_model.eval()
    model = srresnet_model

    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')

    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
    Bicubic_img.save('./results/test_bicubic.jpg')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img.to(device)

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('./results/test_srresnet.jpg')

    print('Done! It takes  {:.3f} seconds.'.format(time.time() - start))
