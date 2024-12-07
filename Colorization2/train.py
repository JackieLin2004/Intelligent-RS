import os
import glob
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Colorization_Model
from my_dataset import ColorData


# 日志函数
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler('./Colorization2.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# 用于计算和存储平均值，通常用来跟踪训练过程中的损失
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


# 用来更新每个损失函数的值
def update_losses(model, loss_meter_dict, losses, count):
    i = 0

    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)
        losses[i].append(loss_meter.avg)
        i += 1

    return losses


# 损失存储器
def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


# 训练函数
def train(model, train_loader, epochs):
    all_loss = [[] for i in range(6)]
    logger = get_logger()
    logger.info("Start training......")

    for epoch in range(epochs):
        loss_meter_dict = create_loss_meters()
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", dynamic_ncols=True) as pbar:
            for data in pbar:
                model.setup_input(data)
                model.optimize()

                all_loss = update_losses(model, loss_meter_dict, all_loss, count=data[0].size(0))

                pbar.set_postfix(
                    loss_D=f"{loss_meter_dict['loss_D'].avg:.4f}",
                    loss_G=f"{loss_meter_dict['loss_G'].avg:.4f}"
                )

        logger.info(f"Epoch {epoch + 1} - "
                    f"loss_D_fake: {loss_meter_dict['loss_D_fake'].avg:.6f}, "
                    f"loss_D_real: {loss_meter_dict['loss_D_real'].avg:.6f}, "
                    f"loss_D: {loss_meter_dict['loss_D'].avg:.6f}, "
                    f"loss_G_GAN: {loss_meter_dict['loss_G_GAN'].avg:.6f}, "
                    f"loss_G_L1: {loss_meter_dict['loss_G_L1'].avg:.6f}, "
                    f"loss_G: {loss_meter_dict['loss_G'].avg:.6f}")

        torch.save({
            'epochs': epochs,
            'model_state_dict': model.state_dict(),
            'losses': all_loss,
        }, os.path.join('./colorization_model.pt'))


# 总运行函数
def run():
    train_path = '../dataset/colorization/train/train_color'
    train_images = glob.glob(train_path + '/*.jpg')

    train_dataset = ColorData(img_paths=train_images, train=1)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=16)

    model = Colorization_Model()

    train(model, train_loader, epochs=100)


if __name__ == '__main__':
    run()
