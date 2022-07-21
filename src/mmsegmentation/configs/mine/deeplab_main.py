_base_ = [
    'deeplabv3plus_r50-d8.py',
    'raildata.py', 'default_runtime.py',
    'schedule_80k.py'
]

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
