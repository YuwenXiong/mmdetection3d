# model settings
_base_ = "./hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py"

voxel_size = [0.15625, 0.15625, 6]
# point_cloud_range = [0, -39.68, -2, 69.12, 39.68, 4]
# point_cloud_range = [0, -39.68, -2, 79.36, 39.68, 4]
point_cloud_range = [0, -40, -2, 72.5, 40.0, 4]

model = dict(
    voxel_layer=dict(
        max_num_points=32,  # max_points_per_voxel
        point_cloud_range=[0, -40, -2, 72.5, 40.0, 4],
        voxel_size=voxel_size,
        max_voxels=(32000, 40000),  # (training, testing) max_voxels
    ),
    voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=3,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -40, -2, 72.5, 40, 4],
    ),
    middle_encoder=dict(type="PointPillarsScatter", in_channels=64, output_shape=[512, 464]),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[0, -40, 0, 72.5, 40, 0]],
            sizes=[[3.9, 1.6, 1.56]],
            # sizes=[[4.2, 2.0, 1.6]],
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
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.1,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=500,
        max_num=100,
    ),
)

file_client_args = dict(backend='disk')

# dataset settings
dataset_type = "KittiDataset"
data_root = "data/kitti/"
class_names = ["Car"]
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "kitti_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3,
        file_client_args=file_client_args),
    classes=class_names,
)

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=3),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="ObjectSample", db_sampler=db_sampler, use_ground_plane=True),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type="PointsRangeFilter", point_cloud_range=[0, -40, -2 - 1.6, 72.5, 40.0, 4 - 1.6]),
    dict(type="ObjectRangeFilter", point_cloud_range=[0, -40, -2 - 1.6, 72.5, 40.0, 4 - 1.6]),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=3),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=[0, -40, -2 - 1.6, 72.5, 40.0, 4 - 1.6]),
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

find_unused_parameters = True
work_dir = "work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car-40ch-adjust_height"
cudnn_benchmark = True
