_base_ = [
    "../_base_/models/hv_pointpillars_secfpn_waymo.py",
    "../_base_/datasets/waymoD2-3d-car.py",
    "../_base_/schedules/schedule_2x.py",
    "../_base_/default_runtime.py",
]

# model settings
model = dict(
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            # ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345]],
            ranges=[[-74.88, -74.88, 0, 74.88, 74.88, 0]],
            # sizes=[[4.7, 2.1, 1.7]],
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
            pos_iou_thr=0.55,
            neg_iou_thr=0.4,
            min_pos_iou=0.4,
            ignore_iof_thr=-1,
        ),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False,
    ),
)
runner = dict(type="EpochBasedRunner", max_epochs=24)
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.01)

# work_dir = "work_dirs/test"
# mp_start_method = "forkserver"
custom_hooks = [dict(type="SyncAWSHook")]
