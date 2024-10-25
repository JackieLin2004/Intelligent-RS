import os
import sys
import json
import random
import logging

import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# 读取和分割数据，获取到训练集和验证集的路径和标签
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    image_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    image_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(image_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in image_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_index = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_index)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_index)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label

# 一次训练函数
def train(model, optimizer, data_loader, device, epoch):
    model.train()   # 模型设定为训练模式
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # 对标签进行平滑处理
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0  # 样本计数器
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum() # 计算正确预测的样本数

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()  # 累计损失

        train_loss = accu_loss.item() / (step + 1)
        train_acc = accu_num.item() / sample_num

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1, train_loss, train_acc)

        if not torch.isfinite(loss):    # 检查损失值是否有限，如果是无穷大或者NAN则终止
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return train_loss, train_acc

# 评估函数，评估验证集
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()    # 模型设定为评估模式

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    # 用于保存所有的真实标签和预测结果
    all_labels = []
    all_preds = []
    all_preds_proba = []  # 存储预测的概率，用于计算AUC

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # 前向传播
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 计算损失
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 收集所有的预测和标签
        all_labels.extend(labels.cpu().numpy())  # 将数据从 GPU 转移到 CPU
        all_preds.extend(pred_classes.cpu().numpy())
        all_preds_proba.extend(torch.softmax(pred, dim=1).cpu().numpy())  # softmax得到类别概率

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    # 计算精确率、召回率、F1 分数和 ROC-AUC
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    # 多分类问题需要加上multi_class='ovo'
    auc = roc_auc_score(all_labels, all_preds_proba, multi_class='ovo')

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, f1, auc

# 日志函数
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler('./AlexNet.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

# 一整次epoch的循环
def train_loop(model, optimizer, train_loader, val_loader, device, epoch, logger):
    # train
    train_loss, train_acc = train(model=model,
                                  optimizer=optimizer,
                                  data_loader=train_loader,
                                  device=device,
                                  epoch=epoch)
    # validate
    val_loss, val_acc, precision, recall, f1, auc = evaluate(model=model,
                                                             data_loader=val_loader,
                                                             device=device,
                                                             epoch=epoch)

    if logger is not None:
        logger.info(f'Epoch: {epoch+1}, '
                    f'Training loss: {train_loss}, '
                    f'Training accuracy: {train_acc}, '
                    f'Validating loss: {val_loss}, '
                    f'Validating accuracy: {val_acc}, '
                    f'Precision: {precision}, '
                    f'Recall: {recall}, '
                    f'f1 score: {f1}, '
                    f'auc: {auc}')

    return val_acc
