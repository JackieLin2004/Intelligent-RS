import torch
import os
from torch.utils.data import DataLoader
from my_dataset import ColorData
from model import Colorization_Model
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Colorization_Model().to(device)

model_path = './colorization_model.pt'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

test_image_paths = [
    './gray/airplane_airplane_562.jpg',
    './gray/airplane_airplane_570.jpg',
    './gray/airplane_airplane_587.jpg',
    './gray/basketball_court_basketball_court_563.jpg',
    './gray/beach_beach_627.jpg',
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


def save_generated_image(model, data, input_image_path):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)  # 获取输入的灰度图和真实彩色图
        model.forward()  # 获取生成的彩色化图像

    fake_color = model.fake_color.detach()  # 获取模型生成的彩色图像
    L = model.L  # 获取亮度通道
    fake_imgs = lab_to_rgb(L, fake_color)  # 将模型输出从 Lab 转换为 RGB

    input_image_name = os.path.basename(input_image_path)
    save_name = os.path.splitext(input_image_name)[0] + '_colorized.jpg'
    save_path = os.path.join('./results', save_name)

    generated_image = fake_imgs[0]
    plt.imsave(save_path, generated_image)
    print(f"Image saved at: {save_path}")


with torch.no_grad():
    for data, input_image_path in zip(test_loader, test_image_paths):
        save_generated_image(model, data, input_image_path)
