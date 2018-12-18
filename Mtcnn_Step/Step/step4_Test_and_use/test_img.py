from PIL import Image
import torch
import numpy as np

img = Image.open("1.jpg")  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=3779x1890 at 0x2957EB8>
img = np.array(img)  # array (1890, 3779, 3)
img = torch.Tensor(img)
img = img.permute((2, 0, 1))  # array (3, 3779, 1890)
img = img.unsqueeze(0)
print(img.shape)
print(img)