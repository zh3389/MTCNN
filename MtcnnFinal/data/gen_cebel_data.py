import os
from PIL import Image
import numpy as np
from tool import utils
import traceback
'''
此文件也是 生成数据的文件 ,建议使用此文件.
'''

anno_src = r"D:\cebela\Anno\list_bbox_celeba.txt"  # 原数据标签文件
img_dir = r"D:\cebela\img_celeba"  # 原数据文件路径

save_path = r"D:\celeba4"  # 保存文件的路径

for face_size in [12]:

    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")  # 生成正样本(图片)文件路径 加 文件名
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")  # 生成负样本(图片)文件路径 加 文件名
    part_image_dir = os.path.join(save_path, str(face_size), "part")  # 生成中等样本(图片)文件路径 加 文件名

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:  # 路径不存在就创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")  # 生成正样本(标签)文件路径 加 文件名
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")  # 生成负样本(标签)文件路径 加 文件名
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")  # 生成中等样本(标签)文件路径 加 文件名

    positive_count = 0  # 正样本计数
    negative_count = 0  # 负样本计数
    part_count = 0  # 中等样本计数

    try:
        positive_anno_file = open(positive_anno_filename, "w")  # 打开 正样本(标签文件) 写入
        negative_anno_file = open(negative_anno_filename, "w")  # 打开 负样本(标签文件) 写入
        part_anno_file = open(part_anno_filename, "w")  # 打开 中等样本(标签文件) 写入

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.strip().split(" ")  # 分割原数据 标签的字符串
                strs = list(filter(bool, strs))  # 将不为空的字符串 依次加入列表
                image_filename = strs[0].strip()  # 将图片名 从标签文件分离出来
                print(image_filename)  # 打印一下文件名
                image_file = os.path.join(img_dir, image_filename)  # 图片路径 = 文件夹路径 + 图片名

                with Image.open(image_file) as img:  # 打开图片
                    img_w, img_h = img.size  # 得到图片的 宽 高
                    x1 = float(strs[1].strip())  # 将 左上角点 x 取出并转换为 float
                    y1 = float(strs[2].strip())  # 将 左上角点 y 取出并转换为 float
                    w = float(strs[3].strip())  # 将 图片的 宽 取出并转换为 float
                    h = float(strs[4].strip())  # 将 图片的 高 取出并转换为 float
                    x2 = float(x1 + w)  # 计算出 右下角点 x 并转换为 float
                    y2 = float(y1 + h)  # 计算出 右下角点 y 并转换为 float

                    '''本文不需要 做人脸识别, 只是做人脸检测, 所以不需要 下面的点 故改为 0 '''
                    px1 = 0  # float(strs[5].strip())
                    py1 = 0  # float(strs[6].strip())
                    px2 = 0  # float(strs[7].strip())
                    py2 = 0  # float(strs[8].strip())
                    px3 = 0  # float(strs[9].strip())
                    py3 = 0  # float(strs[10].strip())
                    px4 = 0  # float(strs[11].strip())
                    py4 = 0  # float(strs[12].strip())
                    px5 = 0  # float(strs[13].strip())
                    py5 = 0  # float(strs[14].strip())

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:  # 去除异常框
                        continue

                    boxes = [[x1, y1, x2, y2]]  # 将框的坐标添加进 boxes 里

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍
                    for _ in range(5):
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.2, w * 0.2)  # 将框的边长 浮动 正负 0.2
                        h_ = np.random.randint(-h * 0.2, h * 0.2)  # 将框的边长 浮动 正负 0.2
                        cx_ = cx + w_  # 计算出中心 x 点坐标
                        cy_ = cy + h_  # 计算出中心 y 点坐标

                        # 让人脸形成正方形，并且让坐标也有少许的偏离(浮动范围 最小边长的0.8 ~ 最大边长的 1.25 之间)
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))  # np.ceil向上取整
                        x1_ = np.max(cx_ - side_len / 2, 0)  # 计算出左上角点 x 坐标, 如果(中心点坐标x - 边长 / 2)小于 0 则取0
                        y1_ = np.max(cy_ - side_len / 2, 0)  # 计算出左上角点 x 坐标, 如果(中心点坐标y - 边长 / 2)小于 0 则取0
                        x2_ = x1_ + side_len  # 计算出右下角 x 点坐标
                        y2_ = y1_ + side_len  # 计算出右下角 y 点坐标

                        crop_box = np.array([x1_, y1_, x2_, y2_])  # 将随机移动的框保存至 np.array

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        ''' 本文不需要 做人脸识别, 只是做人脸检测, 所以不需要 下面的点 故改为 0  '''
                        offset_px1 = 0  # (px1 - x1_) / side_len
                        offset_py1 = 0  # (py1 - y1_) / side_len
                        offset_px2 = 0  # (px2 - x1_) / side_len
                        offset_py2 = 0  # (py2 - y1_) / side_len
                        offset_px3 = 0  # (px3 - x1_) / side_len
                        offset_py3 = 0  # (py3 - y1_) / side_len
                        offset_px4 = 0  # (px4 - x1_) / side_len
                        offset_py4 = 0  # (py4 - y1_) / side_len
                        offset_px5 = 0  # (px5 - x1_) / side_len
                        offset_py5 = 0  # (py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        iou = utils.iou(crop_box, np.array(boxes))[0]  # 计算iou观察图片是 正样本 负样本 中等样本
                        '''接下来就是保存样本了.'''
                        if iou > 0.65:  # 正样本
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif iou > 0.4:  # 部分样本
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2,
                                    offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.3:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                        # 生成负样本
                        _boxes = np.array(boxes)

                    for i in range(5):
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(utils.iou(crop_box, _boxes)) < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
            except Exception as e:
                traceback.print_exc()


    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()
