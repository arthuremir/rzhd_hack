from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import os.path as osp

classes = ('background', 'side_rails', 'main_rails', 'trains')
palette = [[0, 0, 0], [6, 6, 6], [7, 7, 7], [10, 10, 10]]

@DATASETS.register_module()
class RailsDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='_rails.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None
    
@DATASETS.register_module()
class TrainsDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='_trains.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None