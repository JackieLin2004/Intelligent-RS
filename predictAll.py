import os
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image

from GoogleNet.googlenet_model import GoogleNet
from ResNeXt.resnext_model import resnext101_32x8d as ResNeXt101
from AlexNet.alexnet_model import AlexNet
from DenseNet.densenet_model import densenet201
from SwinTransformer.swintransformer_model import swin_base
from VGGNet.vggnet_model import VGG19

from IPV_SRGAN.isrgan_model import Generator as i_Generator
from IPV_SRGAN.utils import *
from SRGAN.srgan_model import Generator
from SRResNet.srresnet_model import SRResNet

from skimage.color import rgb2lab, lab2rgb
from Colorization1.model import ColorizationNet
from Colorization2.model import Colorization_Model
from ESRGAN.models import GeneratorRRDB
from ESRGAN.datasets import denormalize,mean,std
# 模型名称到模型类的映射
MODEL_FENLEI = {
    'GoogleNet': GoogleNet,
    'ResNeXt': ResNeXt101,
    'AlexNet': AlexNet,
    'DenseNet': densenet201,
    'SwinTransformer': swin_base,
    'VGGNet': VGG19,
}

MODEL_CHAOFEN = {
    'IPV_SRGAN': i_Generator,
    'SRGAN': Generator,
    'SRResNet': SRResNet,
}
MODEL_CHAOFEN_key = {
    'IPV_SRGAN': 'generator',
    'SRGAN': 'generator',
    'SRResNet': 'model',
}

def predict_net(image_path, model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    assert os.path.exists(image_path), f"File: '{image_path}' does not exist."
    img = Image.open(image_path)

    # Transform image
    img_transformed = data_transform(img)
    img_tensor = torch.unsqueeze(img_transformed, dim=0)  # Add batch dimension

    # Load class indices
    model_dir = os.path.join('./', model_name)
    json_path = os.path.join('class_indices.json')
    assert os.path.exists(json_path), f"File: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Load the model
    if model_name not in MODEL_FENLEI:
        raise ValueError(f"Unsupported model: {model_name}")

    model_class = MODEL_FENLEI[model_name]
    model = model_class(num_classes=45).to(device)
    weights_path = os.path.join(model_dir, 'best_model.pth')
    assert os.path.exists(weights_path), f"File: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # Perform inference
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    result_text = f"class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla].numpy():.3f}"

    plt.imshow(img)
    plt.title(result_text)
    result_filename = f'result_{model_name}_{os.path.basename(image_path)}'
    result_path = os.path.join('results', model_name, result_filename)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)  # 确保结果目录存在
    plt.savefig(result_path)
    plt.close()

    return result_text, result_path, class_indict


def predict_chaofen(image_path, model_name):
    if model_name == 'ESRGAN':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义生成器模型并加载训练好的模型
        generator = GeneratorRRDB(channels=3, filters=64, num_res_blocks=23).to(device)
        model_dir = os.path.join('./', model_name)
        checkpoint_model = os.path.join(model_dir, 'generator_best.pth')  # 模型路径
        assert os.path.exists(checkpoint_model), f"File: '{checkpoint_model}' does not exist."
        generator.load_state_dict(torch.load(checkpoint_model, map_location=device))
        generator.eval()

        # 定义图像预处理和后处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 加载并准备输入图像
        assert os.path.exists(image_path), f"File: '{image_path}' does not exist."
        image = Image.open(image_path).convert('RGB')
        image = image.resize((128, 128))  # 确保输入图像为 128x128
        image_tensor = Variable(transform(image)).to(device).unsqueeze(0)

        # 超分辨率处理
        with torch.no_grad():
            sr_image = generator(image_tensor)  # 生成超分辨率图像
            sr_image = torch.nn.functional.interpolate(sr_image, size=(256, 256), mode='bilinear',
                                                       align_corners=False)  # 调整输出大小为 256x256
            sr_image = denormalize(sr_image).cpu()  # 反归一化

        # 保存输出图像
        os.makedirs(f'./results/{model_name}/', exist_ok=True)
        fn = os.path.basename(image_path)  # 获取文件名
        result_path = f'./results/{model_name}/sr-{fn}'
        save_image(sr_image, result_path)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型参数
        large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
        small_kernel_size = 3  # 中间层卷积的核大小
        n_channels = 64  # 中间层通道数
        n_blocks = 16  # 残差模块数量
        scaling_factor = 2  # 放大比例

        checkpoint_path = f"./{model_name}/checkpoint_{model_name}.pth"
        # 加载模型SRGAN
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_class = MODEL_CHAOFEN[model_name]
        model_key = MODEL_CHAOFEN_key[model_name]
        generator = model_class(large_kernel_size=large_kernel_size,
                                small_kernel_size=small_kernel_size,
                                n_channels=n_channels,
                                n_blocks=n_blocks,
                                scaling_factor=scaling_factor)
        generator = generator.to(device)
        generator.load_state_dict(checkpoint[model_key])

        generator.eval()
        model = generator

        # 加载图像
        img = Image.open(image_path, mode='r')
        img = img.convert('RGB')

        # 双线性上采样
        Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
        Bicubic_img.save(f'./results/{model_name}/result_bicubic_{os.path.basename(image_path)}')

        # 图像预处理
        lr_img = convert_image(img, source='pil', target='imagenet-norm')
        lr_img.unsqueeze_(0)

        # 转移数据至设备
        lr_img = lr_img.to(device)

        # 模型推理
        with torch.no_grad():
            sr_img = model(lr_img).squeeze(0).cpu().detach()
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
            result_path = f'./results/{model_name}/result_{model_name}_{os.path.basename(image_path)}'
            sr_img.save(result_path)

    return result_path


def predict_color(image_path, model_name):
    if model_name == 'Colorization1':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载颜色化模型
        model_dir = os.path.join('./', model_name)
        weights_path = os.path.join(model_dir, 'model_best_params.pkl')
        assert os.path.exists(weights_path), f"File: '{weights_path}' does not exist."

        color_model = ColorizationNet().to(device)
        color_model.load_state_dict(torch.load(weights_path, map_location=device))
        color_model.eval()

        # 加载输入的图像并转换为灰度图像
        assert os.path.exists(image_path), f"File: '{image_path}' does not exist."

        # 加载并转换图像为灰度图（1通道）
        img = Image.open(image_path).convert('L')  # 'L' 模式表示灰度图像
        original_size = img.size  # 获取原始图像大小

        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整大小为模型输入大小
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度并移到设备

        # 将灰度图转换为伪彩色的 RGB 图像
        img_rgb = img.convert('RGB')  # 将灰度图转为 RGB 图像

        # 将图像从 RGB 转换为 LAB 颜色空间
        img_rgb_np = np.array(img_rgb)  # 转换为 numpy 数组
        img_lab = rgb2lab(img_rgb_np)  # 使用 RGB 图像进行 LAB 转换
        img_lab = (img_lab + 128) / 255  # 归一化到 [0, 1]
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))  # 转换为 [2, H, W] 的形状

        # 运行模型并获取输出
        with torch.no_grad():
            _, output = color_model(img_tensor, img_tensor)  # 传入灰度图进行颜色化
            color_img = torch.cat((img_tensor, output), dim=1)  # 拼接原图和输出图像

            # 将结果从张量转换为 numpy 以便进一步处理
            color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))[0]

        # 后处理 LAB 图像为 RGB
        color_img[:, :, 0] = color_img[:, :, 0] * 100  # 还原 L 通道
        color_img[:, :, 1:3] = color_img[:, :, 1:3] * 255 - 128  # 还原 AB 通道
        color_img = lab2rgb(color_img.astype(np.float64))

        # 调整生成的图像大小与原图相同
        result_image = Image.fromarray((color_img * 255).astype(np.uint8))
        result_image = result_image.resize(original_size, Image.LANCZOS)  # 调整为原图大小

        # 保存结果
        os.makedirs(f'./results/{model_name}/', exist_ok=True)
        result_path = f'./results/{model_name}/result_{os.path.basename(image_path)}'
        result_image.save(result_path)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载颜色化模型
        model_dir = os.path.join('./', model_name)
        weights_path = os.path.join(model_dir, 'colorization_model.pt')
        assert os.path.exists(weights_path), f"File: '{weights_path}' does not exist."

        color_model = Colorization_Model().to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        color_model.load_state_dict(checkpoint['model_state_dict'])
        color_model.eval()

        # 加载输入的图像并转换为灰度图像
        assert os.path.exists(image_path), f"File: '{image_path}' does not exist."

        # 加载并转换图像为 RGB 图像
        img = Image.open(image_path).convert('RGB')  # 加载 RGB 图像
        original_size = img.size  # 获取原始图像大小

        # 预处理图像：调整大小并转换为张量
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整大小为模型输入大小
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度并移到设备

        # 将 RGB 转换为 Lab 颜色空间
        img_lab = rgb2lab(img_tensor.permute(0, 2, 3, 1).cpu().numpy())
        L = img_lab[:, :, :, 0]  # 亮度通道
        ab = img_lab[:, :, :, 1:3]  # 色度通道

        # 归一化
        L = (L / 50.0) - 1.0  # 归一化到 [-1, 1]
        ab = ab / 110.0  # 归一化到 [-1, 1]

        # 转换为张量
        L = torch.from_numpy(L).unsqueeze(1).to(device)  # 增加通道维度
        ab = torch.from_numpy(ab.transpose((0, 3, 1, 2))).to(device)  # 转换为 [B, 2, H, W]

        # 模拟 DataLoader 的输出，构造一个包含灰度图和色度图的元组
        data = (L, ab)

        # 运行模型并获取输出
        with torch.no_grad():
            color_model.setup_input(data)  # 设置输入数据
            color_model.forward()  # 生成彩色化图像

        # 获取生成的彩色图像和亮度通道
        fake_color = color_model.fake_color.detach()  # 获取模型生成的彩色图像
        L = color_model.L  # 获取亮度通道

        # 将 Lab 图像转换为 RGB
        def lab_to_rgb(L, ab):
            L = (L + 1.) * 50.
            ab = ab * 110.
            Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
            rgb_imgs = []
            for img in Lab:
                img_rgb = lab2rgb(img)
                rgb_imgs.append(img_rgb)
            return np.stack(rgb_imgs, axis=0)

        fake_imgs = lab_to_rgb(L, fake_color)  # 将模型输出从 Lab 转换为 RGB

        # 调整生成的图像大小与原图相同
        result_image = Image.fromarray((fake_imgs[0] * 255).astype(np.uint8))
        result_image = result_image.resize(original_size, Image.LANCZOS)  # 调整为原图大小

        # 保存结果
        os.makedirs(f'./results/{model_name}/', exist_ok=True)
        result_path = f'./results/{model_name}/result_{os.path.basename(image_path)}'
        result_image.save(result_path)


    return result_path

