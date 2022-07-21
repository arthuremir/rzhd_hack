
# dataset settings
dataset_type = 'RailsDataset'
# data_root = '../data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (1280, 720)
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        img_ratios=[0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], # RAILS AUG
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),

    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root='/mmsegmentation/data_prepared/train',
        img_dir='images',
        ann_dir='mask',
        pipeline=train_pipeline,
        split = '/mmsegmentation/data_prepared/splits/train.txt'),
    val=dict(
        type=dataset_type,
        data_root='/mmsegmentation/data_prepared/train',
        img_dir='images',
        ann_dir='mask',
        pipeline=valid_pipeline,
        split = '/mmsegmentation/data_prepared/splits/val.txt'),
    test=dict(
        type=dataset_type,
        data_root='/mmsegmentation/data_prepared/test',
        img_dir='images',
        # ann_dir='labels',
        pipeline=test_pipeline,
        split = '/mmsegmentation/data_prepared/splits/test.txt',
        ))