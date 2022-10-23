# model settings

_base_ = [
    "../_base_/models/waabi_two_stage_kitti.py",
    # "../_base_/datasets/kitti-3d-3class.py",
    "../_base_/datasets/pandaset-3d-car.py",
    "../_base_/schedules/seg_cosine_50e.py",
    "../_base_/default_runtime.py",
]


point_cloud_range = [0, -40, -2, 80.0, 40, 4]
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

# dataset settings
class_names = ["Car"]
dataset_type = "PandasetDataset"
data_root = "data/pandaset/"
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "pandaset_car_bottom_infos_train.pkl",
    rate=1.0,
    prepare=dict(filter_by_min_points=dict(Car=1)),  # filter_by_difficulty=[-1],
    sample_groups=dict(Car=15),
    classes=class_names,
)

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=3, use_dim=3),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    # dict(type="ObjectSample", db_sampler=db_sampler, use_ground_plane=False),
    # dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.3490658504, 0.3490658504],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[2.0, 2.0, 0.5],
    ),
    dict(type="PointsRangeFilter", point_cloud_range=[0, -40, -2 - 0.4, 80.0, 40, 4 - 0.4]),
    dict(type="ObjectRangeFilter", point_cloud_range=[0, -40, -2 - 0.4, 80.0, 40, 4 - 0.4]),
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
            dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0]),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=[0, -40, -2 - 0.4, 80.0, 40, 4 - 0.4]),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]

optimizer = dict(type="AdamW", lr=0.0001, weight_decay=1e-4)
runner = dict(type="EpochBasedRunner", max_epochs=400)

data = dict(
    train=dict(type="RepeatDataset", times=2, dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names),
)

runner = dict(max_epochs=40)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)

work_dir = "work_dirs/waabi_two_stage_pandaset-3d-car-run5_data_fix"

# work_dir = "work_dirs/pandaset_overfit"

find_unused_parameters = True
