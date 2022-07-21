# optimizer
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=99999)
evaluation = dict(interval=500, metric='mIoU', pre_eval=True, save_best='mIoU')
