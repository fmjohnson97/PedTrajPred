import numpy as np
from PIL import Image, ImageEnhance
from glob import glob
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

files =glob('frames/frame*.png')
images=[]
for f in files:
    im=Image.open(f)
    enhancer = ImageEnhance.Contrast(im)
    factor = 0.75  # increase contrast
    im = enhancer.enhance(factor)
    enhancer = ImageEnhance.Sharpness(im)
    factor = 4
    im = enhancer.enhance(factor)
    # enhancer = ImageEnhance.Brightness(im)
    # factor = 1.25  # brightens the image
    # im = enhancer.enhance(factor)
    # plt.imshow(im)
    # plt.show()
    images.append(np.array(im).astype(np.float32))

images=np.stack(images)
image=np.sum(images,axis=0)
image/=len(images)
plt.imshow(image.astype(np.uint8))
plt.show()