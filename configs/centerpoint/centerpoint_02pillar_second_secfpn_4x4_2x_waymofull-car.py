_base_ = [
    "../_base_/datasets/waymoD2-3d-car.py",
    "../_base_/models/centerpoint_02pillar_second_secfpn_waymofull.py",
    "../_base_/schedules/schedule_2x.py",
    "../_base_/default_runtime.py",
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-75, -75, -2.0, 75, 75, 4.0]
# For nuScenes we usually do 10-class detection
class_names = ["Car"]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(
        bbox_coder=dict(pc_range=point_cloud_range[:2], code_size=7),
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), _delete_=True),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(point_cloud_range=point_cloud_range, code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])),
)


optimizer = dict(lr=0.001 / 2)
evaluation = dict(interval=24)
checkpoint_config = dict(interval=1)
find_unused_parameters = True
# work_dir = "work_dirs/centerpoint-pandaset-3d-car-real-aug"
