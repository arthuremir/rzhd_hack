from PIL import Image
from glob import glob
import imagesize
import numpy as np
import cv2
import shutil

imgs = glob('work_dirs/upernet_convnext_xlarge_v1_mad/pred_maps_57500_tta2/*')

res = set()
for path in imgs:

    spath = '../../data/test/images/' + path.split('/')[-1]
    
    im = np.array(Image.open(path).resize(imagesize.get(spath), resample=Image.NEAREST))
    
    res.add(im.shape)
    # resmask = im[..., 0]
    # resmask[(im == (0, 255, 0)).all(axis=-1)] = 7
    # resmask[(im == (0, 0, 255)).all(axis=-1)] = 6
    # resmask[(im == (0, 0, 0)).all(axis=-1)] = 10
    # resmask[(im == (255, 0, 0)).all(axis=-1)] = 0
    # im = np.stack([resmask, resmask, resmask], axis=2)

    cv2.imwrite(path, im)
print(res)
shutil.make_archive('work_dirs/upernet_convnext_xlarge_v1_mad/solution_57500_tta2', 'zip', 'work_dirs/upernet_convnext_xlarge_v1_mad/pred_maps_57500_tta2/')