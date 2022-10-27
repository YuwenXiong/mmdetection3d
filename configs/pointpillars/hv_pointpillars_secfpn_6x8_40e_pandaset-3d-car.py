# model settings
_base_ = [
    "../_base_/models/hv_pointpillars_waabi_secfpn_pandaset.py",
    "../_base_/datasets/pandaset-3d-car.py",
    "../_base_/schedules/cyclic_40e.py",
    "../_base_/default_runtime.py",
]

# point_cloud_range = [0, -39.68, -2, 79.36, 39.68, 4]
point_cloud_range = [0, -40, -2, 80.0, 40.0, 4]
data_root = "data/pandaset_sim512/"
# data_root = "data/pandaset/"
class_names = ["Car"]
file_client_args = dict(backend="disk")
model = dict(
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type="AlignedAnchor3DRangeGenerator",
            # ranges=[[0, -39.68, 0, 79.36, 39.68, 0]],
            ranges=[[0, -40.0, 0, 80.0, 40.0, 0]],
            # sizes=[[4.2, 2.0, 1.6]],
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
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=0.5,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.1,
        score_thr=0.0,
        min_bbox_size=0,
        nms_pre=500,
        max_num=100,
    ),
)


train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=3, use_dim=3),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    # dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type="PointsRangeFilter", point_cloud_range=[0, -40, -2 - 0.4, 80.0, 40.0, 4 - 0.4]),
    dict(type="ObjectRangeFilter", point_cloud_range=[0, -40, -2 - 0.4, 80.0, 40.0, 4 - 0.4]),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=3, use_dim=3),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=[0, -40, -2 - 0.4, 80.0, 40.0, 4 - 0.4]),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]

data = dict(
    train=dict(type="RepeatDataset", times=2, dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names),
)
# data = dict(
#     train=dict(dataset=dict(pipeline=train_pipeline, classes=class_names)),
#     val=dict(pipeline=test_pipeline, classes=class_names),
#     test=dict(pipeline=test_pipeline, classes=class_names),
# )


# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001 / 4
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
runner = dict(max_epochs=40)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=2)
checkpoint_config = dict(interval=2)


find_unused_parameters = True
work_dir = "work_dirs/hv_pointpillars_secfpn_6x8_80e_pandaset-3d-car-binarynewvoxel_height_fix_kitti_anchor_sim512_data_fix4"
cudnn_benchmark = True
