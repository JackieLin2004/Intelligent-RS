import os
import traceback
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from img_folder import TrainImageFolder
from model import ColorizationNet


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler('./Colorization1.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

img_size = 256
original_transform = transforms.Compose([
    transforms.Resize(int(img_size * 1.143)),
    transforms.CenterCrop(img_size),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

have_cuda = torch.cuda.is_available()
epochs = 10

data_dir = '../dataset/colorization_/train'
train_set = TrainImageFolder(data_dir, original_transform)
train_set_size = len(train_set)
train_set_classes = train_set.classes
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
color_model = ColorizationNet()
if os.path.exists('./model_best_params.pkl'):
    color_model.load_state_dict(torch.load('model_best_params.pkl'))
if have_cuda:
    color_model.cuda()
optimizer = optim.Adadelta(color_model.parameters())

logger = get_logger()
logger.info("Start training...")

def train(epoch, best_loss):
    color_model.train()
    epoch_loss = 0.0
    try:
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", ncols=100) as pbar:
            for batch_idx, (data, classes) in enumerate(train_loader):
                original_img = data[0].unsqueeze(1).float()
                img_ab = data[1].float()

                if have_cuda:
                    original_img = original_img.cuda()
                    img_ab = img_ab.cuda()
                    classes = classes.cuda()

                original_img = Variable(original_img)
                img_ab = Variable(img_ab)
                classes = Variable(classes)
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播
                class_output, output = color_model(original_img, original_img)
                # 损失 = 均方误差 + 交叉熵损失
                ems_loss = torch.pow((img_ab - output), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod()
                cross_entropy_loss = 1/300 * F.cross_entropy(class_output, classes)
                loss = ems_loss + cross_entropy_loss
                # 累积损失
                epoch_loss += loss.item()
                # 反向传播
                ems_loss.backward(retain_graph=True)
                cross_entropy_loss.backward()
                # 更新优化器
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch: {epoch}, '
                    f'Training loss: {avg_epoch_loss}')

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(color_model.state_dict(), './model_best_params.pkl')
            logger.info("Best model saved!")

    except Exception:
        exceptionFile = open('./exception.txt', 'w')
        exceptionFile.write(traceback.format_exc())
        exceptionFile.close()

    return best_loss


if __name__ == '__main__':
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        best_loss = train(epoch, best_loss)
