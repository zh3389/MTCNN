from SubjectCode import Three_Net
from SubjectCode import tool
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw


class use_Net:
    def __init__(self):
        '''
        初始化网络和权重
        '''
        self.pnet = Three_Net.PNet()
        self.pnet = torch.load("save/12")
        self.pnet.eval()
        self.rnet = Three_Net.RNet()
        self.rnet = torch.load("save/24")
        self.rnet.eval()
        self.onet = Three_Net.ONet()
        self.onet = torch.load("save/48")
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

        x1 = int(source_box_out[0] + ow * R_O_box_out[0, 0])
        y1 = int(source_box_out[1] + oh * R_O_box_out[0, 1])
        x2 = int(source_box_out[2] + ow * R_O_box_out[0, 2])
        y2 = int(source_box_out[3] + oh * R_O_box_out[0, 3])
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
            # source_img.crop([int(i[0]), int(i[1]), int(i[2]), int(i[3])]).show()
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
            # self.face_out = self.face_out.cpu()
            # self.box_out = self.box_out.cpu()
            self.face_out = self.face_out.squeeze(0)  # torch.Size([2, 940, 1884])
            self.box_out = self.box_out.squeeze(0)  # torch.Size([4, 940, 1884])
            face_index = torch.nonzero(torch.gt(self.face_out[1, :, :], 0.65))  # torch.Size([344, 2]) x和y的位置(卷积后的图)
            for index in face_index:
                boxes.append(
                    self.p_box(index, self.box_out[:, index[0], index[1]], self.face_out[1, index[0], index[1]], scale))
            # 0.702
            scale *= 0.702
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_w_h = min(_w, _h)
        # self.imageDraw(boxes)
        return tool.nms(torch.Tensor(boxes).detach().numpy(), 0.3)  # 低于0.5的保留

        # self.face_out = self.face_out.permute((0, 2, 3, 1)).squeeze(0)  # 将输出转置, 降维  torch.Size([940, 1884, 2])
        # self.box_out = self.box_out.permute((0, 2, 3, 1)).squeeze(0)  # 将输出转置, 降维  torch.Size([940, 1884, 4])
        # self.face_index = torch.lt(self.face_out[:, :, 0], 0.5)  # 取出置信度高的框的索引  torch.Size([485, 2])
        # self.face_out, self.box_out = (self.face_out[self.face_index])[:, 1], self.box_out[self.face_index]  # 将输出的值更新为需要的值

    # cls0.6  nms 0.5
    def R_use(self, source_box_list, img):  # torch.Size([421, 24, 24, 3])  (421, 5)
        img_box_list = self.crop_img(source_box_list, img, resize_img=24)
        self.r_box_list = img_box_list.permute((0, 3, 1, 2))
        self.r_face_out, self.r_box_out = self.rnet(self.r_box_list)
        self.r_face_index = torch.nonzero(torch.gt(self.r_face_out[:, 1], 0.9))
        self.boxes = []
        for index in self.r_face_index:
            self.boxes.append(
                self.r_o_box(torch.Tensor(source_box_list[index]), self.r_box_out[index], self.r_face_out[index, 1]))
        # self.imageDraw(self.boxes)
        return tool.nms(torch.Tensor(self.boxes), 0.2)

    # cls0.6  nms 0.7
    def O_use(self, source_box_list, img):  # torch.Size([421, 24, 24, 3])  (421, 5)
        img_box_list = self.crop_img(source_box_list, img, resize_img=48)
        self.o_box_list = img_box_list.permute((0, 3, 1, 2))
        self.o_face_out, self.o_box_out = self.onet(self.o_box_list)
        self.o_face_index = torch.nonzero(torch.gt(self.o_face_out[:, 1], 0.9999))
        self.boxes = []
        for index in self.o_face_index:
            self.boxes.append(
                self.r_o_box(torch.Tensor(source_box_list[index]), self.o_box_out[index], self.o_face_out[index, 1]))
        return tool.nms(torch.Tensor(self.boxes), 0.7, isBase=False)

    def imageDraw(self, img, box_list):
        '''
        :param box_list: 传入盒子的列表
        :return: 返回画好框的图
        '''
        # img = Image.open(img_path)  # 将图片路径和图片名path.join组合并打开该图片
        img_copy = img.copy()
        imgDraw = ImageDraw.Draw(img_copy)  # 将img用画板的方式打开
        for i in box_list:
            imgDraw.rectangle((int(i[0]), int(i[1]), int(i[2]), int(i[3])),
                              outline='red')  # 进行绘图(rectangle矩形) 输入左上点右下点坐标 线的颜色
        img_copy.show()  # 显示图片


if __name__ == '__main__':
    '''
    执行此文件 可观察网络 框人脸的具体情况(使用网络)
    需要修改的位置为 159行的 图片位置
    如果图片太大电脑运行不了 将下两行缩放图片的注释取消掉即可
    '''
    test_Net = use_Net()
    img_path = "test_pic/4.jpg"
    img = Image.open(img_path)  # 打开一张图片
    # w, h = img.size
    # img = img.resize((w//3, h//3))
    P_Net_box_list = test_Net.P_use(img)  # 将图片传入P网络 的到p网络出来的原图框
    # P_Net_show = test_Net.imageDraw(img, P_Net_box_list)  # 将P网络的框画在图片上展示一下

    R_Net_box_list = test_Net.R_use(tool.convert_to_square(P_Net_box_list), img)
    # R_Net_show = test_Net.imageDraw(img, R_Net_box_list)  # 将R网络的框画在图片上展示一下

    O_Net_box_list = test_Net.O_use(tool.convert_to_square(R_Net_box_list), img)
    O_Net_show = test_Net.imageDraw(img, O_Net_box_list)
