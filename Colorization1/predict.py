import os

import torch
from torch.autograd import Variable
from skimage.color import lab2rgb
from model import ColorizationNet
from img_folder import ValImageFolder
import numpy as np
import matplotlib.pyplot as plt


data_dir = "./test"
have_cuda = torch.cuda.is_available()

val_set = ValImageFolder(data_dir)
val_set_size = len(val_set)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

color_model = ColorizationNet()
color_model.load_state_dict(torch.load('./model_best_params.pkl', map_location=torch.device('cpu')))
if have_cuda:
    color_model.cuda()


def predict():
    color_model.eval()

    # 创建保存灰度图像和彩色图像的目录
    if not os.path.exists('./gray'):
        os.makedirs('./gray')
    if not os.path.exists('./colorimg'):
        os.makedirs('./colorimg')

    i = 0
    for data, _ in val_loader:
        original_img = data[0].unsqueeze(1).float()
        gray_name = './gray/' + str(i) + '.jpg'
        for img in original_img:
            pic = img.squeeze().numpy()
            pic = pic.astype(np.float64)
            plt.imsave(gray_name, pic, cmap='gray')
        w = original_img.size()[2]
        h = original_img.size()[3]
        scale_img = data[1].unsqueeze(1).float()
        if have_cuda:
            original_img, scale_img = original_img.cuda(), scale_img.cuda()

        original_img, scale_img = Variable(original_img, volatile=True), Variable(scale_img)
        _, output = color_model(original_img, scale_img)
        color_img = torch.cat((original_img, output[:, :, 0:w, 0:h]), 1)
        color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
        for img in color_img:
            # 归一化处理
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            color_name = './colorimg/' + str(i) + '.jpg'
            plt.imsave(color_name, img)
            i += 1


if __name__ == '__main__':
    predict()
