import torch
import os
from torch.utils.data import DataLoader
from my_dataset import ColorData
from model import Colorization_Model
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import numpy as np

# 准备模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Colorization_Model().to(device)

# 加载模型
model_path = './colorization_model.pt'  # 替换为你的模型路径
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

test_image_paths = [
    '../dataset/colorization/test/test_grayscale/airplane_airplane_561.jpg',
]

test_dataset = ColorData(img_paths=test_image_paths, train=0)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)


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


# 只保存生成的彩色图像
def save_generated_image(model, data):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)  # 获取输入的灰度图和真实彩色图
        model.forward()  # 获取生成的彩色化图像

    fake_color = model.fake_color.detach()  # 获取模型生成的彩色图像
    L = model.L  # 获取亮度通道
    fake_imgs = lab_to_rgb(L, fake_color)  # 将模型输出从 Lab 转换为 RGB

    # 保存生成的彩色图像
    generated_image = fake_imgs[0]  # 只取第一张图像（batch size 为 1）
    save_path = os.path.join('./colorized_image.png')  # 保存路径
    plt.imsave(save_path, generated_image)
    print(f"Image saved at: {save_path}")


# 测试过程
with torch.no_grad():
    for data in test_loader:
        save_generated_image(model, data)
