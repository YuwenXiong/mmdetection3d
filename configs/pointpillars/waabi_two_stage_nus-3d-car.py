# model settings

_base_ = [
    "../_base_/models/waabi_two_stage_kitti.py",
    # "../_base_/datasets/kitti-3d-3class.py",
    "../_base_/datasets/nus-3d-car.py",
    "../_base_/schedules/seg_cosine_50e.py",
    "../_base_/default_runtime.py",
]


point_cloud_range = [-50, -50, -2, 50.0, 50, 4]
model = dict(
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


find_unused_parameters = True
