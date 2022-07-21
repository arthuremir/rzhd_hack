from PIL import Image
from glob import glob
import imagesize
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import os

imgs_trains = sorted(glob('work_dirs/upernet_convnext_xlarge_trains/pred_maps_trains_tta/*'))
imgs_rails = sorted(glob('work_dirs/upernet_convnext_xlarge_rails/pred_maps_rails_tta/*'))

save_dir = 'final_predict/'
os.makedirs(save_dir, exist_ok=True)

for img_trains, img_rails in tqdm(zip(imgs_trains, imgs_rails)):

    spath = '../../data/test/images/' + img_trains.split('/')[-1]
    
    im_trains = np.array(Image.open(img_trains).resize(imagesize.get(spath), resample=Image.NEAREST))
    im_rails = np.array(Image.open(img_rails).resize(imagesize.get(spath), resample=Image.NEAREST))
    
    resmask = im_rails[..., 0]
    resmask[(im_trains == (10, 10, 10)).all(axis=-1)] = 10

    im = np.stack([resmask, resmask, resmask], axis=2)

    cv2.imwrite(save_dir + img_trains.split('/')[-1], im)
    
shutil.make_archive('solution', 'zip', save_dir)