import numpy as np
from PIL import Image
import os
from Mtcnn_Step.Step.step2_Tool_and_data.data.IOU import iou


class Data_production:
    '''
    :objective: 制作数据, 将标签框随机浮动制作为 (12*12, 24*24, 48*48)的标签框 (正样本 iou>0.65, 负样本 iou<0.3, 中等样本 0.65 > iou > 0.4)
    :input:     img, label
    :return:    resize为:(12*12, 24*24, 48*48)的图 和 (正样本, 负样本, 中等样本)
    '''

    def __init__(self):
        '''
        :objective: Initialization source path
        :param self.source_img_path: Initialization path
        :param self.source_label_path: Initialization path
        :param self.makedir(): if not have dir, so makedir.
        '''
        self.source_img_path = "D:/celeba/img_celeba/"  # 源图片路径
        self.source_label_path = "D:/celeba/tag/list_bbox_celeba.txt"  # 原标签路径

        self.folder_path = self.makedir()  # 文件夹路径的列表
        print(self.folder_path)

    def makedir(self):  # 创建存储路径
        '''
        :objective if not have dir, so makedir
        :return: dir_list: [[12(positive)], [12(negative)], [12(medium)],
                            [24(positive)], [24(negative)], [24(medium)],
                            [48(positive)], [48(negative)], [48(medium)],]
        '''
        # 生成样本路径
        # 12*12
        filepath_12 = "D:/celeba/data/12/"
        if not os.path.exists(filepath_12):
            os.makedirs(filepath_12)
        # 24*24
        filepath_24 = "D:/celeba/data/24/"
        if not os.path.exists(filepath_24):
            os.makedirs(filepath_24)
        # 48*48
        filepath_48 = "D:/celeba/data/48/"
        if not os.path.exists(filepath_48):
            os.makedirs(filepath_48)

        positive_samples = "positive"  # 正样本文件夹名
        negative_samples = "negative"  # 负样本文件夹名
        medium_samples = "medium"  # 中等样本文件夹名

        # 创建 12*12样本的文件夹路径(如果没有的话)
        if not os.path.exists(filepath_12 + positive_samples):
            os.makedirs(filepath_12 + positive_samples)
        if not os.path.exists(filepath_12 + negative_samples):
            os.makedirs(filepath_12 + negative_samples)
        if not os.path.exists(filepath_12 + medium_samples):
            os.makedirs(filepath_12 + medium_samples)
        # 创建 24*24样本的文件夹路径(如果没有的话)
        if not os.path.exists(filepath_24 + positive_samples):
            os.makedirs(filepath_24 + positive_samples)
        if not os.path.exists(filepath_24 + negative_samples):
            os.makedirs(filepath_24 + negative_samples)
        if not os.path.exists(filepath_24 + medium_samples):
            os.makedirs(filepath_24 + medium_samples)
        # 创建 48*48样本的文件夹路径(如果没有的话)
        if not os.path.exists(filepath_48 + positive_samples):
            os.makedirs(filepath_48 + positive_samples)
        if not os.path.exists(filepath_48 + negative_samples):
            os.makedirs(filepath_48 + negative_samples)
        if not os.path.exists(filepath_48 + medium_samples):
            os.makedirs(filepath_48 + medium_samples)
        dir_list = [[filepath_12 + positive_samples], [filepath_12 + negative_samples], [filepath_12 + medium_samples],
                    [filepath_24 + positive_samples], [filepath_24 + negative_samples], [filepath_24 + medium_samples],
                    [filepath_48 + positive_samples], [filepath_48 + negative_samples], [filepath_48 + medium_samples]]
        return dir_list

    def arranging_information(self, img_message):  # 整理信息
        '''
        :param img_message: 输入图片信息, str[name, x, y, w, h]
        :return: name, xywh, x1y1x2y2, center_x_y, max_label_w_h
        :return: [str(name), int(x), int(y), int(w), int(h)], label_window, [int(center_x), int(center_y)], int(max_label_w_h)
        '''
        name1, label_x1, label_y1, label_w1, label_h1 = img_message  # 将 图片的名字 左上角坐标, 图片的宽和高取出(取出是字符串)
        label_x1, label_y1, label_w1, label_h1 = int(label_x1), int(label_y1), int(label_w1), int(
            label_h1)  # 将取出的字符串转换为整型
        max_label_w_h = max(label_w1, label_h1)  # 获得标签框宽高的最大值
        label_x2, label_y2 = label_x1 + label_w1, label_y1 + label_h1  # 计算出图片的 右下角的坐标
        label_window = np.array([label_x1, label_y1, label_x2, label_y2])  # 将标签框的坐标放入label_window
        center_x, center_y = label_x1 + (label_w1 / 2), label_y1 + (label_h1 / 2)  # 计算图片中心点
        return [name1, [label_x1, label_y1, label_w1, label_h1], label_window,
                np.array([center_x, center_y], dtype=int), max_label_w_h]

    def random_window(self, label_xywh, label_window, center, max_label_w_h, img, mobility_ratio = 0.08):  # 生成随机的框
        '''
        :param label_xywh: 输入标签框的 [x1, y1, w, h]
        :param label_window: 输入标签框的 [x1, y1, x2, y2]
        :param center: 输入标签框的中心点 [x, y]
        :param max_label_w_h: 输入标签框 w 或 h 中的最大值
        :param img: 输入图片
        :return: random_label_list [x1, y1, x2, y2]
        '''
        img_w, img_h = img.size
        label_list = []
        # 将标签框的中心点移位 宽高的 10% 左右
        center_x_, center_y_ = center[0] + np.random.randint(int(-label_xywh[2] * mobility_ratio), int(
            label_xywh[2] * mobility_ratio)), center[1] + np.random.randint(int(-label_xywh[3] * mobility_ratio),
                                                                            int(label_xywh[3] * mobility_ratio))

        x1_ = int(label_window[0] + (center_x_ - center[0]))  # 计算当前正方形框的 x1坐标
        y1_ = int(label_window[1] + (center_y_ - center[1]))  # 计算当前正方形框的 y1坐标
        x2_ = int(x1_ + max_label_w_h)
        y2_ = int(y1_ + max_label_w_h)

        # # 生成负样本 随机框 随机边长
        # x1_ = int(np.random.randint(img_w))
        # y1_ = int(np.random.randint(img_h))
        # x2_ = int(x1_ + np.random.randint(48, img_w))
        # y2_ = int(y1_ + (x2_ - x1_))

        if x1_ > 0 and y1_ > 0 and x2_ < img_w and y2_ < img_h:
            label_list.append([x1_, y1_, x2_, y2_])  # 将计算出来的框的坐标添加进列表
        # count5 += 1
        return np.array(label_list, dtype=int)

    def calculated_offset(self, max_w_h, label_window, label_list):  # 计算每个框的偏移量
        '''
        :param max_w_h: 生成的随机正方形标签框的边长
        :param label_window: 标签框的坐标 (x1, y1, x2, y2)
        :param label_list: 随机批量生成框的坐标 ([x1, y1, x2, y2], [x1, y1, x2, y2], ...)
        :return: numpy.array([offset_list])  # 返回一个计算好偏移量的数组[[0.1, 0.2, -0.1, -0.2],[], ...]
        '''
        label_list = label_list.reshape(-1, 4)
        offset_list = []
        if label_list.shape[0] == 0:
            return np.array(offset_list)
        for i in label_list:
            x1_ = (label_window[0] - i[0]) / max_w_h
            y1_ = (label_window[1] - i[1]) / max_w_h
            x2_ = (label_window[2] - i[2]) / max_w_h
            y2_ = (label_window[3] - i[3]) / max_w_h
            offset_list.append([x1_, y1_, x2_, y2_])
        return np.array(offset_list)

    def split_data(self, label_window, random_label_list):  # 拆分数据为: 正样本 负样本 中等样本
        '''
        :param label_window: 标签框的坐标 (x1, y1, x2, y2)
        :param random_label_list: 随机截取原图的标签框
        :return: 正样本, 负样本, 中等样本
        '''
        random_label_list = random_label_list.reshape(-1, 4)  # 将输入的random_label_list reshape成(批次, 4)
        label_transfer = iou(label_window, random_label_list)  # 存放暂时的 iou
        label_positive = random_label_list[np.where(label_transfer > 0.7)]
        label_medium = random_label_list[np.where(label_transfer[np.where(label_transfer > 0.3)] < 0.65)]
        label_negative = random_label_list[np.where(label_transfer < 0.3)]
        return np.array([label_positive, label_negative, label_medium])

    def picture_processing(self, img_resize=12):
        '''
        :objective: 使用标签框 和 原图 随机浮动造出我需要的数据
        :return: 输出到对应文件夹数据
        '''
        self.positive_img_count = 0  # 正样本计数
        self.negative_img_count = 0  # 负样本计数
        self.medium_img_count = 0  # 中等样本计数
        for _ in range(15):
            with open(self.source_label_path) as f:  # 打开标签文件获得 文件名 和 标签框的左上角点 和 宽高
                for i, j in enumerate(f.readlines()):  # 使用枚举类 遍历文件的每一行数据
                    if i < 2:  # 去掉前两行
                        continue
                    img_message = j.split()  # 将字符串分割后 返回一个列表 [img_name, x1, y1, w, h]
                    # [str(name), int(x), int(y), int(w), int(h)], label_window, [int(center_x), int(center_y)], int(max_label_w_h)
                    name, label_xywh, label_window, center, max_label_w_h = self.arranging_information(img_message)
                    with Image.open(self.source_img_path + name) as img:  # 打开一张图片
                        img_w, img_h = img.size  # 获得图片的原宽和高
                        # 过滤掉 人脸太小的框 和 超出图片范围的框
                        if (label_xywh[2] < 48 and label_xywh[3] < 48) or label_window[0] < 0 or label_window[1] < 0 or \
                                label_window[2] < 0 or label_window[3] < 0:
                            continue
                        random_label_list = self.random_window(label_xywh, label_window, center, max_label_w_h, img, mobility_ratio=0.06)  # 得到随机的标签框数组(返回整数的随机边框)
                        label_list_offset = self.calculated_offset(max_label_w_h, label_window, random_label_list)  # 将偏移量计算出来返回数组(返回每个的偏移量)
                        label_positive, label_negative, label_medium = self.split_data(label_window, random_label_list)  # 将随机裁剪的样本传入 得到正样本,负样本,中等样本(随机框坐标)
                        if random_label_list.shape[0] == 0:
                            continue
                        face_crop = img.crop(random_label_list[0])

                        # 正样本保存方法
                        # if label_positive.shape[0] != 0:
                        #     self.positive_img_count += 1
                        #     face_crop = face_crop.resize((img_resize, img_resize))
                        #     face_crop.save("D:/celeba/data/{}/positive/{}.jpg".format(img_resize, self.positive_img_count))
                        #     with open("D:/celeba/data/{}/positive_label.txt".format(img_resize), "a") as f:
                        #         a = "{0}.jpg {1} {2} {3} {4} {5} \n".format(self.positive_img_count, str(1), label_list_offset[0][0], label_list_offset[0][1], label_list_offset[0][2], label_list_offset[0][3])
                        #         f.write(a)
                        #     if self.positive_img_count % 100 == 0:
                        #         print("positive:{}".format(self.positive_img_count))

                        # 负样本保存方法
                        # if label_negative.shape[0] != 0:
                        #     self.negative_img_count += 1
                        #     face_crop = face_crop.resize((img_resize, img_resize))
                        #     face_crop.save("D:/celeba/data/{}/negative/{}.jpg".format(img_resize, self.negative_img_count))
                        #     with open("D:/celeba/data/{}/negative_label.txt".format(img_resize), "a") as f:
                        #         a = "{}.jpg 0 0 0 0 0 \n".format(self.negative_img_count)
                        #         f.write(a)
                        #     if self.negative_img_count % 100 == 0:
                        #         print("negative:{}".format(self.negative_img_count))

                        # 中等样本保存方法
                        # if label_medium.shape[0] != 0:
                        #     self.medium_img_count += 1
                        #     face_crop = face_crop.resize((img_resize, img_resize))
                        #     face_crop.save("D:/celeba/data/{}/medium/{}.jpg".format(img_resize, self.medium_img_count))
                        #     with open("D:/celeba/data/{}/medium_label.txt".format(img_resize), "a") as f:
                        #         a = "{0}.jpg {1} {2} {3} {4} {5} \n".format(self.medium_img_count, str(2), label_list_offset[0][0], label_list_offset[0][1], label_list_offset[0][2], label_list_offset[0][3])
                        #         f.write(a)
                        #     if self.medium_img_count % 100 == 0:
                        #         print("medium:{}".format(self.medium_img_count))


if __name__ == '__main__':
    data_production = Data_production()
    data_production.picture_processing(img_resize=12)
