# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [0, -40, -2 - 0.4, 70.4, 40, 4 - 0.4]
point_cloud_range = [0, -40, -2 - 0.4, 80, 40, 4 - 0.4]
# point_cloud_range=[0, -39.68, -2, 79.36, 39.68, 4],
# For pandaset we usually do 3-class detection
# class_names = ['Car', 'Pedestrian', 'Cyclist']
class_names = ["Car"]
dataset_type = "PandasetDataset"
data_root = "data/pandaset/"
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend="disk")
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))


train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=3, use_dim=3, file_client_args=file_client_args),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type="ObjectNoise",
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0.5],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-0.78539816, 0.78539816],
    # ),
    # dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=3, use_dim=3, file_client_args=file_client_args),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=3, use_dim=3, file_client_args=file_client_args),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["points"]),
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "pandaset_car_bottom_infos_train.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            filter_empty_gt=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "pandaset_car_bottom_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "pandaset_car_bottom_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=1, pipeline=eval_pipeline)
