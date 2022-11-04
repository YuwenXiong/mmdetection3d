# model settings

_base_ = [
    "../_base_/models/waabi_two_stage_kitti.py",
    # "../_base_/datasets/kitti-3d-3class.py",
    "../_base_/datasets/waymoD2_front-3d-car.py",
    "../_base_/schedules/schedule_2x.py",
    "../_base_/default_runtime.py",
]


# point_cloud_range = [-75, -75, -2, 75, 75, 4]
model = dict(
    backbone=dict(
        type="ResNet",
        in_channels=40,
        groups_in_stem=1,
        stride_in_stem=2,
        channels_in_stem=[64, 64, 64],
        blocks_per_stage=[3, 3, 3],
        block_type="BasicBlock",
        # base_channels=64,
        # specify_channels=True,
        channels_per_stage=[96, 128, 160],
        strides_per_stage=[2, 2, 2],
        dilations_per_stage=[1, 1, 1],
        out_indices=[1, 2, 3],
        norm_type="BN",
    ),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type="MaxIoUAssigner",
            iou_calculator=dict(type="BboxOverlapsNearest3D"),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
)

# dataset settings

optimizer = dict(type="AdamW", lr=0.002, weight_decay=1e-4)
runner = dict(type="EpochBasedRunner", max_epochs=24)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=24)
checkpoint_config = dict(interval=1)

# work_dir = "work_dirs/waabi_two_stage_pandaset-3d-car-run5_data_fix"
# work_dir = "work_dirs/pandaset_overfit"
custom_hooks = [dict(type="SyncAWSHook")]
find_unused_parameters = True
