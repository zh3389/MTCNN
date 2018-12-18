from PIL import Image
import numpy as np

img = Image.open('3.jpg')
print(img.size)
img.crop([10, 10, 100, 100]).resize((48, 48)).show()
img = np.array(img.crop([10, 10, 100, 100]).resize((48, 48)))
print(img.shape)
img.show()
