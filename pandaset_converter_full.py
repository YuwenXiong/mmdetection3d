# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import mmcv
import numpy as np
import os
from os import path as osp
from pandaset_splits import splits_dir
import pandaset as ps
import re
from glob import glob
import pickle
from pytorch3d import transforms
from pytorch3d.transforms.transform3d import Transform3d
import torch
from tools.data_converter.create_gt_database import GTDatabaseCreater, create_groundtruth_database

pandaset_categories = ["Car", "Pedestrian", "Cyclist"]

# This maps raw dataset categories with the corresponding categories used in training
# This map can be incomplete. In case a category is not present, the category
# for training is the same as the raw dataset category
PANDASET_CLASS_MAPPING = {
    "Car": "Car",
    "Pickup Truck": "Car",
    "Medium-sized Truck": "Car",
    "Semi-truck": "Car",
    "Towed Object": "Car",
    "Other Vehicle - Construction Vehicle": "Car",
    "Other Vehicle - Uncommon": "Car",
    "Other Vehicle - Pedicab": "Car",
    "Emergency Vehicle": "Car",
    "Bus": "Car",
    # "Motorcycle": "Cyclist",
    # "Bicycle": "Cyclist",
    # "Pedestrian": "Pedestrian",
    # "Pedestrian with Object": "Pedestrian",
}


def get_split(splits_folder: str, version: str = "0.1"):
    """Get the split given the version, or return the latest one"""

    splits = {}
    files = sorted(glob(os.path.join(splits_folder, version, "*.txt")))
    for split_file in files:
        split_name = re.sub(r"\.txt$", "", split_file.split("/")[-1])
        seqs = open(split_file, "r").readlines()
        seqs = [s.rstrip("\n") for s in seqs]
        splits[split_name] = seqs
    return splits


def pose_dict_to_numpy(pose):
    """
    Conert pandaset pose dict to a numpy vector in order to pass it through the network
    """
    pose_np = [
        pose["position"]["x"],
        pose["position"]["y"],
        pose["position"]["z"],
        pose["heading"]["w"],
        pose["heading"]["x"],
        pose["heading"]["y"],
        pose["heading"]["z"],
    ]

    return pose_np


def pose_numpy_to_dict(pose):
    """
    Conert pandaset pose dict to a numpy vector in order to pass it through the network
    """
    pose_dict = {
        "position": {"x": pose[0], "y": pose[1], "z": pose[2]},
        "heading": {"w": pose[3], "x": pose[4], "y": pose[5], "z": pose[6]},
    }

    return pose_dict


def create_pandaset_infos(root_path, out_path, info_prefix, split_version="0.1", max_sweeps=10):
    """Create info file of pandaset dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    print(root_path)
    pandaset = ps.DataSet(root_path)

    splits = get_split(splits_dir(), split_version)
    # filter existing scenes.
    available_scene_names = pandaset.sequences()
    train_scenes_names = splits["training"]
    val_scenes_names = splits["validation"]

    train_scenes_names = list(filter(lambda x: x in available_scene_names, train_scenes_names))
    val_scenes_names = list(filter(lambda x: x in available_scene_names, val_scenes_names))

    train_scenes = [pandaset[s] for s in train_scenes_names]
    val_scenes = [pandaset[s] for s in val_scenes_names]

    print("train scene: {}, val scene: {}".format(len(train_scenes_names), len(val_scenes_names)))

    train_pandaset_infos = _fill_trainval_infos(train_scenes, max_sweeps=max_sweeps)
    val_pandaset_infos = _fill_trainval_infos(val_scenes, max_sweeps=max_sweeps)

    metadata = dict(split_version=split_version)

    print("train sample: {}, val sample: {}".format(len(train_pandaset_infos), len(val_pandaset_infos)))
    data = dict(infos=train_pandaset_infos, metadata=metadata)
    info_path = osp.join(out_path, "{}_infos_train.pkl".format(info_prefix))
    # info_path = osp.join(out_path,
    #                         '{}_infos_train_overfit.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_pandaset_infos
    info_val_path = osp.join(out_path, "{}_infos_val.pkl".format(info_prefix))
    # info_val_path = osp.join(out_path,
    #                             '{}_infos_val_overfit.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)

    # create_groundtruth_database(
    #     'PandasetDataset',
    #     root_path,
    #     info_prefix,
    #     f'{out_path}/{info_prefix}_infos_train.pkl',
    #     relative_path=False,
    #     mask_anno_path='instances_train.json',
    #     with_mask=False)


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    sample_timestamp = sample["timestamp"]
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop("utime")  # useless
    pos = last_pose.pop("pos")
    rotation = last_pose.pop("orientation")
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def _fill_trainval_infos(scenes, max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    pandaset_infos = []

    for scene in mmcv.track_iter_progress(scenes):
        # scene.load_()
        scene.lidar._load_poses()
        # scene.load_timestamps()
        # scene.load_lidar()
        data_dir = scene._directory
        seq_id = os.path.basename(data_dir)
        # for sample in mmcv.track_iter_progress(nusc.sample):

        for frame_idx in range(1, len(scene.lidar.poses)):
            info = {
                "sequence": seq_id,
                "frame_idx": frame_idx,
                # "lidar_path": os.path.join(data_dir, "lidar", ("{:02d}.pkl".format(frame_i
                # dx))),
                "lidar_path": os.path.join(data_dir, "lidar", ("{:03d}.npy".format(frame_idx))),
                "cuboids_path": os.path.join(data_dir, "annotations", "cuboids", ("{:02d}.pkl".format(frame_idx))),
                # "cuboids_path": os.path.join(data_dir, "annotations", "cuboids", ("{:03d}.npy".format(frame_idx))),
                # 'semseg_path': os.path.join(data_dir, 'annotations', 'semseg', ("{:02d}.pkl.gz".format(frame_idx))),
                # 'classes_path': os.path.join(data_dir, 'annotations', 'semseg', "classes.json"),
                # "cams": dict(),
                # "timestamp": scene.timestamps.data[frame_idx],
            }

            # camera_types = [
            #     "front_camera",
            #     "front_left_camera",
            #     "front_right_camera",
            #     "left_camera",
            #     "right_camera",
            #     "back_camera",
            # ]
            # for cam in camera_types:
            #     camera = scene.camera[cam]
            #     camera._load_poses()
            #     camera._load_intrinsics()

            #     cam_intrinsic = np.eye(3)
            #     cam_intrinsic[0, 0] = camera.intrinsics.fx
            #     cam_intrinsic[1, 1] = camera.intrinsics.fy
            #     cam_intrinsic[0, 2] = camera.intrinsics.cx
            #     cam_intrinsic[1, 2] = camera.intrinsics.cy

            #     cam_path = os.path.join(data_dir, "camera", cam, ("{:02d}.jpg".format(frame_idx)))
            #     cam_pose = pose_dict_to_numpy(scene.camera[cam].poses[frame_idx])
            #     pos, quat = torch.tensor(cam_pose[:3]), torch.tensor(cam_pose[3:])
            #     # camera_pose = lidar_pose_dict_to_pose3d(pose).inverse()
            #     cam2world = (
            #         (
            #             Transform3d()
            #             .rotate(
            #                 transforms.quaternion_to_matrix(transforms.quaternion_invert(quat)),
            #             )
            #             .translate(pos[0], pos[1], pos[2])  # type: ignore
            #         )
            #         .get_matrix()
            #         .T[..., 0]
            #     )
            #     cam_info = {"cam_intrinsic": cam_intrinsic, "data_path": cam_path, "cam_pose": cam2world}

            #     # 'lidar2ego_translation': cs_record['translation'],
            #     # 'lidar2ego_rotation': cs_record['rotation'],
            #     # 'ego2global_translation': pose_record['translation'],
            #     # 'ego2global_rotation': pose_record['rotation'],

            #     info["cams"].update({cam: cam_info})

            # obtain annotation
            cuboids_path = info["cuboids_path"]
            annotations = pickle.load(open(cuboids_path, "rb"))
            # try:
            #     annotations2 = np.load(
            #         os.path.join(data_dir, "annotations", "cuboids", ("{:03d}.npy".format(frame_idx))), allow_pickle=True
            #     ).item()
            # except:
            #     print(cuboids_path)
            # # annotations = scene.cuboids.data

            # annotations = {k: v for k, v in annotations.items() if k in annotations2}
            # for k in annotations.keys():
            #     annotations[k]['num_pts'] = annotations2[k][0]['num_pts']

            locs = np.array(
                [label["position"] for _, label in annotations.items() if label["cuboids_sensor_id"] != 1]
            ).reshape(-1, 3)
            sdv_pose = scene.lidar.poses[frame_idx]

            sdv_pose_np = pose_dict_to_numpy(sdv_pose)
            pos, quat = torch.tensor(sdv_pose_np[:3]), torch.tensor(sdv_pose_np[3:])
            ego2world = (
                (
                    Transform3d()
                    .rotate(
                        transforms.quaternion_to_matrix(transforms.quaternion_invert(quat)),
                    )
                    .translate(pos[0], pos[1], pos[2])  # type: ignore
                )
                .get_matrix()
                .T.numpy()[..., 0]
            )

            locs_ego = ps.geometry.lidar_points_to_ego(locs, sdv_pose)
            # lidar_data = scene.lidar.data[frame_idx]
            # points = np.stack([np.array(lidar_data['x']), np.array(lidar_data['y']), np.array(lidar_data['z'])], axis=-1)
            # points_ego = ps.geometry.lidar_points_to_ego(points, sdv_pose)

            dims = np.array(
                [label["dimension"] for _, label in annotations.items() if label["cuboids_sensor_id"] != 1]
            ).reshape(-1, 3)
            yaws = np.array(
                [label["yaw"] for _, label in annotations.items() if label["cuboids_sensor_id"] != 1]
            ).reshape(-1, 1)
            valid_flag = np.array(
                [label["num_pts"] > 0 for _, label in annotations.items() if label["cuboids_sensor_id"] != 1],
                dtype=bool,
            ).reshape(-1)

            names = [label["label"] for _, label in annotations.items() if label["cuboids_sensor_id"] != 1]
            masks = []
            for i in range(len(names)):
                if names[i] in PANDASET_CLASS_MAPPING:
                    names[i] = PANDASET_CLASS_MAPPING[names[i]]
                    masks.append(True)
                else:
                    masks.append(False)

            names = np.array(names)
            masks = np.array(masks)

            try:
                locs = locs[masks]
            except:
                import ipdb; ipdb.set_trace()

            locs_ego = locs_ego[masks]
            dims = dims[masks]
            names = names[masks]
            valid_flag = valid_flag[masks]
            yaws = yaws[masks]

            yaxis_points_from_pose = ps.geometry.lidar_points_to_ego(np.array([[0, 0, 0], [0, 1.0, 0]]), sdv_pose)
            yaxis_from_pose = yaxis_points_from_pose[1, :] - yaxis_points_from_pose[0, :]
            # rotation angle in rads of the y axis around thz z axis
            zrot_world_to_ego = np.arctan2(-yaxis_from_pose[0], yaxis_from_pose[1])
            ego_yaws = yaws + zrot_world_to_ego

            #### switch to FLU
            gt_boxes = np.concatenate(
                # [locs_ego[:, [1]], -locs_ego[:, [0]], locs_ego[:, [2]], dims[:, [1, 0, 2]], ego_yaws], axis=1
                [
                    locs_ego[:, [1]],
                    -locs_ego[:, [0]],
                    locs_ego[:, [2]] - dims[:, [2]] / 2,
                    dims[:, [1, 0, 2]],
                    ego_yaws,
                ],
                axis=1,
            )
            # gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], -yaws + np.pi / 2], axis=1)
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["num_lidar_pts"] = np.array(
                [label["num_pts"] for _, label in annotations.items() if label["cuboids_sensor_id"] != 1]
            )[masks]
            info["valid_flag"] = valid_flag
            info["sdv_pose"] = sdv_pose
            info["ego2world"] = ego2world
            # info['points_ego'] = np.concatenate([points_ego[:, [1]], -points_ego[:, [0]], points_ego[:, [2]]], axis=1)

            assert len(gt_boxes) == len(names) == len(valid_flag) == len(info["num_lidar_pts"])

            pandaset_infos.append(info)

    return pandaset_infos


if __name__ == "__main__":
    # create_pandaset_infos("/mnt/data/pandaset_pnp_npt_cam", "/mnt/data/pandaset_pnp_npt_cam", "pandaset")
    create_pandaset_infos(
        "./data/pandaset_sim512",
        "./data/pandaset_sim512",
        "pandaset_car_bottom",
        "0.1",
    )

    # create_pandaset_infos("/mnt/data/pandaset_pnp_npt_cam", "/mnt/data/pandaset_pnp_npt_cam", "pandaset", "pnp_sim_0.1")
    # create_pandaset_infos("/mnt/data/pandaset_pnp_npt_cam", "/mnt/data/pandaset_pnp_npt_cam", "pandaset", "overfit")
