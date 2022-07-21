from glob import glob
from random import shuffle
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import imagesize
import cv2

img_list = glob('data/train/mask/*')

for path in tqdm(img_list):
    
    img = Image.open(path).resize((1280, 720), resample=Image.NEAREST)
    mask = np.array(Image.open(path))
    mask[mask == 6] = 1
    mask[mask == 7] = 2
    mask[mask == 10] = 3
    
    cv2.imwrite(path, mask)
    
    mask_path = path.replace('images/', 'mask/').replace('data', 'data_prepared')
    os.makedirs('/'.join(mask_path.split('/')[:-1]), exist_ok=True)
    mask.save(mask_path)
    
    img_path = path.replace('data', 'data_prepared').replace('.png', '.jpg')
    os.makedirs('/'.join(img_path.split('/')[:-1]), exist_ok=True)
    img.save(img_path)
    

img_list = glob('data/test/*')

for path in tqdm(img_list):
    
    img = Image.open(path).resize((1280, 720), resample=Image.NEAREST)

    img_path = path.replace('data', 'data_prepared').replace('.png', '.jpg')
    os.makedirs('/'.join(img_path.split('/')[:-1]), exist_ok=True)
    img.save(img_path)