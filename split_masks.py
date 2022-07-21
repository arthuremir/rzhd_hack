from glob import glob
from random import shuffle
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import cv2


imgs = glob('data_prepared/train/mask/*')
print(len(imgs))

for i, img in tqdm(enumerate(imgs)):
    if '1ch' in img or 'rails' in img or 'trains' in img:
        continue

    im = np.array(Image.open(img))

    im[im == 3] = 0

    cv2.imwrite(img.replace('.png', '_rails.png'), im)
    
    
for i, img in tqdm(enumerate(imgs)):
    if '1ch' in img or 'rails' in img or 'trains' in img:
        continue

    im = np.array(Image.open(img))

    im[im != 3] = 0

    cv2.imwrite(img.replace('.png', '_trains.png'), im)
