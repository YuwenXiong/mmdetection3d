_base_ = [
    '../_base_/models/hv_second_secfpn_kitti.py',
    '../_base_/datasets/pandaset-3d-car.py', '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py'
]
point_cloud_range = [0, -40, -2, 70.4, 40, 4]
model = dict(
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, 0, 70.4, 40.0, 0]],
            sizes=[[4.2, 2.0, 1.6]],
            # sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=500,
        max_num=100))
