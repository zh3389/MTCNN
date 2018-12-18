from PIL import Image
from PIL import ImageDraw
import os

# 观察样本的情况

IMG_Dir = r"D:\celeba\img_celeba"  # IMG dir
TAG_Dir = r"D:\celeba\tag"  # TAG dir

img = Image.open(os.path.join(IMG_Dir, "032317.jpg"))  # 将图片路径和图片名path.join组合并打开该图片
imgDraw = ImageDraw.Draw(img)  # 将img用画板的方式打开
imgDraw.rectangle((121, 132, 121 + 114, 132 + 158), outline='red')  # 进行绘图(rectangle矩形) 输入左上点右下点坐标 线的颜色
img.show()  # 显示图片
