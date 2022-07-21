_base_ = [
    'upernet_r50.py',
    'raildata_769x769.py', 'default_runtime.py',
    'schedule_80k.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
