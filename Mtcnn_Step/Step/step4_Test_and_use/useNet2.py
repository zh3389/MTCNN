from Mtcnn_Step.Step.step4_Test_and_use import Three_Net
from Mtcnn_Step.Step.step4_Test_and_use import tool
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
import os


class use_Net:
    def __init__(self):
        '''
        初始化网络和权重
        '''
        self.pnet = Three_Net.PNet()
        self.pnet = torch.load(r"D:\python_prjs\Study\MTCNN\Step\step3_Net_and_train\save\12")
        self.pnet.eval()
        self.rnet = Three_Net.RNet()
        self.rnet = torch.load(r"D:\python_prjs\Study\MTCNN\Step\step3_Net_and_train\save\24")
        self.rnet.eval()
        self.onet = Three_Net.ONet()
        self.onet = torch.load(r"D:\python_prjs\Study\MTCNN\Step\step3_Net_and_train\save\48")
        self.rnet.eval()

    def p_box(self, start_index, box_out, face_out_cls, scale, stride=2, side_len=12):
        '''
        :param start_index: 网络输出框的索引
        :param box_out: 网络输出框的偏移量
        :param face_out_cls: 网络输出的置信度
        :param scale: 图片缩小的比例
        :param stride: 整个卷积的步长
        :param side_len: 整个卷积的窗口大小
        :return: 计算好的原图框的坐标 和置信度
        '''
        _x1 = (start_index[1].float() * stride) / scale  # 计算原图框的x1
        _y1 = (start_index[0].float() * stride) / scale  # 计算原图框的y1
        _x2 = (start_index[1].float() * stride + side_len) / scale  # 计算原图框的x2
        _y2 = (start_index[0].float() * stride + side_len) / scale  # 计算原图框的y2

        ow = _x2 - _x1  # 计算原图框的 w
        oh = _y2 - _y1  # 计算原图框的 h

        _offset = box_out  # 将图像的偏移量取出来[x1, y1, x2, y2]用来计算原框的位置
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, face_out_cls]

    def r_o_box(self, source_box_out, R_O_box_out, face_out_cls):
        ow = source_box_out[2] - source_box_out[0]  # 计算原图框的 w
        oh = source_box_out[3] - source_box_out[1]  # 计算原图框的 h

        x1 = source_box_out[0] + ow * R_O_box_out[0, 0]
        y1 = source_box_out[1] + oh * R_O_box_out[0, 1]
        x2 = source_box_out[2] + ow * R_O_box_out[0, 2]
        y2 = source_box_out[3] + oh * R_O_box_out[0, 3]
        return [x1, y1, x2, y2, face_out_cls]

    def crop_img(self, box_list, source_img, resize_img=24):
        '''
        :param box_list: 建议框
        :param source_img: 原图
        :param resize_img: resize大小
        :return: some_img
        '''
        img_list = []
        for i in box_list:
            crop_img_one = source_img.crop([int(i[0]), int(i[1]), int(i[2]), int(i[3])]).resize(
                (resize_img, resize_img))
            img_list.append(np.array(crop_img_one, dtype=np.float32) / 255. - 0.5)
        return torch.Tensor(img_list)

    def P_use(self, source_img):
        '''
        :param source_img: 需要框脸的图片
        :return: 返回计算好的所有框
        '''
        boxes = []  # 用来存放计算好的框
        img = source_img  # 将图片拷贝一份
        w, h = img.size
        min_w_h = min(w, h)
        scale = 1  # 图片缩放比例
        while min_w_h > 12:
            self.img = torch.Tensor(np.array(img, dtype=np.float) / 255. - 0.5)  # 打开一张图片 转为数组, 归一化
            self.img = self.img.permute((2, 0, 1)).unsqueeze(0)  # 将图片转置(hwc -> chw) 并升维 (加一个批次)
            self.face_out, self.box_out = self.pnet(
                self.img)  # torch.Size([1, 2, 940, 1884])  torch.Size([1, 4, 940, 1884])
            self.face_out = self.face_out.squeeze(0)  # torch.Size([2, 940, 1884])
            self.box_out = self.box_out.squeeze(0)  # torch.Size([4, 940, 1884])
            face_index = torch.nonzero(torch.gt(self.face_out[1, :, :], 0.6))  # torch.Size([344, 2]) x和y的位置(卷积后的图)
            for index in face_index:
                boxes.append(
                    self.p_box(index, self.box_out[:, index[0], index[1]], self.face_out[1, index[0], index[1]], scale))
            scale *= 0.702
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_w_h = min(_w, _h)
        return tool.nms(torch.Tensor(boxes).detach().numpy(), 0.3)

        # self.face_out = self.face_out.permute((0, 2, 3, 1)).squeeze(0)  # 将输出转置, 降维  torch.Size([940, 1884, 2])
        # self.box_out = self.box_out.permute((0, 2, 3, 1)).squeeze(0)  # 将输出转置, 降维  torch.Size([940, 1884, 4])
        # self.face_index = torch.lt(self.face_out[:, :, 0], 0.5)  # 取出置信度高的框的索引  torch.Size([485, 2])
        # self.face_out, self.box_out = (self.face_out[self.face_index])[:, 1], self.box_out[self.face_index]  # 将输出的值更新为需要的值

    def R_O_use(self, net, img_box_list, source_box_list):  # torch.Size([421, 24, 24, 3])  (421, 5)
        self.R_O_Net = net
        self.box_list = img_box_list.permute((0, 3, 1, 2))
        self.R_O_face_out, self.R_O_box_out = self.R_O_Net(self.box_list)
        self.face_index = torch.nonzero(torch.gt(self.R_O_face_out[:, 1], 0.3))
        self.boxes = []
        for index in self.face_index:
            self.boxes.append(self.r_o_box(torch.Tensor(source_box_list[index]), self.R_O_box_out[index], self.R_O_face_out[index, 1]))
        # print(torch.Tensor(self.boxes).detach.numpy())
        return tool.nms(torch.Tensor(self.boxes), 0.2)


    def imageDraw(self, box_list):
        '''
        :param box_list: 传入盒子的列表
        :return: 返回画好框的图
        '''
        img = Image.open("2.jpg")  # 将图片路径和图片名path.join组合并打开该图片
        imgDraw = ImageDraw.Draw(img)  # 将img用画板的方式打开
        for i in box_list:
            imgDraw.rectangle((int(i[0]), int(i[1]), int(i[2]), int(i[3])),
                              outline='red')  # 进行绘图(rectangle矩形) 输入左上点右下点坐标 线的颜色
        img.show()  # 显示图片


if __name__ == '__main__':
    test_Net = use_Net()
    img = Image.open("2.jpg")  # 打开一张图片

    P_Net_box_list = test_Net.P_use(img)  # 将图片传入P网络 的到p网络出来的原图框
    P_Net_show = test_Net.imageDraw(P_Net_box_list)  # 将P网络的框画在图片上展示一下
    P_Net_crop_img = test_Net.crop_img(P_Net_box_list, img, resize_img=24)  # 将图片从原图抠下来 resize 准备传入下一个网络

    print("p box shape:", P_Net_box_list.shape)
    print("p img shape:", P_Net_crop_img.shape)

    R_Net_box_list = test_Net.R_O_use(test_Net.rnet, P_Net_crop_img, P_Net_box_list)
    R_Net_show = test_Net.imageDraw(R_Net_box_list)
    R_Net_crop_img = test_Net.crop_img(R_Net_box_list, img, resize_img=48)

    print("r box shape:", R_Net_box_list.shape)
    print("r img shape:", R_Net_crop_img.shape)

    O_Net_box_list = test_Net.R_O_use(test_Net.onet, R_Net_crop_img, R_Net_box_list)
    O_Net_show = test_Net.imageDraw(O_Net_box_list)

    print("o box shape:", O_Net_box_list.shape)