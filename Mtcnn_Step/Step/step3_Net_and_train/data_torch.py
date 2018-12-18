from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, path):
        '''
        :param path: 传入数据路径
        :return  : 将样本添加到同一个列表里
        '''
        self.path = path  # 传入样本数据路径
        self.dataset = []  # 定义一个列表用于存放所有的样本(正, 负, 一般)
        self.dataset.extend(open(os.path.join(path, "positive_label.txt")).readlines())  # 将样本按行添加到 self.dataset里
        self.dataset.extend(open(os.path.join(path, "negative_label.txt")).readlines())  # 将样本按行添加到 self.dataset里
        self.dataset.extend(open(os.path.join(path, "medium_label.txt")).readlines())  # 将样本按行添加到 self.dataset里

    def __getitem__(self, item):
        '''
        :param item: 传入需要加载到序列里的 数据
        :return:  返回一个可迭代的对象
        '''
        message = self.dataset[item].split()  # 将文本拆分 为 ['name', '置信度', 'x1偏移', 'y1偏移', 'x2偏移', 'y2偏移']
        img_path = os.path.join(self.path, message[0])  # 得到图片完整路径
        confidence = torch.Tensor([int(message[1])])  # 得到置信度
        offset = torch.Tensor([float(message[2]), float(message[3]), float(message[4]), float(message[5])])  # 得到四个偏移量 float
        img_data = torch.Tensor(np.array((Image.open(img_path)), dtype=np.float64) / 255. - 0.5)  # 将图片数据读出来 并归一化
        img_data = img_data.permute(2, 0, 1)  # 将数据从 HWC 转换为 CHW
        # img_data = img_data.view(3, )
        return img_data, confidence, offset  # 返回 图片数据, 置信度, 偏移量

    def __len__(self):
        '''
        :return: 返回 dataset的长度
        '''
        return len(self.dataset)


if __name__ == '__main__':
    dataset = FaceDataset("D:/celeba/train_data/12/")
    print(dataset[0])
