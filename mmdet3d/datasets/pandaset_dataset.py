# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
import torch


def transform(pts: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
    """Homogeneous points transformation
    Args:
        pts: NxD
        tr: (D+1)x(D+1) or DxD
    Return:
        pts_tr: NxD or Nx(D-1)
    Examples:
        >>> intrinsic = torch.eye(3)
        >>> intrinsic[1,2] = -100 # crop
        >>> intrinsic[0,0] = 2 # scale
        >>> camera_pts = torch.rand(100,3)*200
        >>> transform(camera_pts, intrinsic).shape
            torch.Size([100, 2])
    """
    # pts = torch.Tensor(pts) if not torch.is_tensor(pts) else pts
    # tr = torch.Tensor(tr) if not torch.is_tensor(tr) else tr
    if pts.shape[-1] + 1 == tr.shape[-1]:  # cat with 1 to make it homogeneous
        pts = torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1)
    pts_hom_tr = pts @ (tr.T)
    pts_tr = pts_hom_tr[..., :-1] / pts_hom_tr[..., -1, None]
    return pts_tr


@DATASETS.register_module()
class PandasetDataset(Custom3DDataset):
    r"""Pandaset Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    PANDASET_CLASS_MAPPING = {
        "Car": "Car",
        "Pickup Truck": "Car",
        "Medium-sized Truck": "Car",
        "Semi-truck": "Car",
        "Towed Object": "Car",
        "Motorcycle": "Cyclist",
        "Other Vehicle - Construction Vehicle": "Car",
        "Other Vehicle - Uncommon": "Car",
        "Other Vehicle - Pedicab": "Car",
        "Emergency Vehicle": "Car",
        "Bus": "Car",
        "Bicycle": "Cyclist",
        "Pedestrian": "Pedestrian",
        "Pedestrian with Object": "Pedestrian",
    }

    CLASSES = ("Car", "Pedestrian", "Cyclist")

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        bev_size=(200, 200),
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        # self.eval_version = eval_version
        # from nuscenes.eval.detection.config import config_factory
        # self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format="pkl")
        # data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data["infos"]
        data_infos = data_infos[:: self.load_interval]
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        input_dict = dict(
            pts_filename=info["lidar_path"],
            timestamp=info["timestamp"],
            sdv_pose=info["sdv_pose"],
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info["cams"].items():
                # print (index, cam_type)
                # print (cam_info['cam_pose'])
                # print (cam_info['cam_intrinsic'])

                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                cam2world = cam_info["cam_pose"]
                intrinsic = cam_info["cam_intrinsic"]
                cam2screen = np.eye(4)
                cam2screen[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

                world2screen = torch.tensor(cam2screen, dtype=torch.float32) @ torch.tensor(cam2world).inverse()
                ego2world = info["ego2world"]
                ego2screen = world2screen @ torch.tensor(ego2world, dtype=torch.float32)
                lidar2img_rts.append(ego2screen.numpy())

                # import open3d as o3d
                # import cv2; import imageio
                # # intri = torch.tensor(cfg['frames'][idx]['intrinsic'], dtype=torch.float32)
                # # o3d_pcd = o3d.io.read_point_cloud("/home/wangjk/Downloads/pcd.ply")
                # # verts = torch.tensor(np.asarray(o3d_pcd.points), dtype=torch.float32)[::10] # verts in obj
                # import pandas as pd
                # data = pd.read_pickle(info['lidar_path'], compression="gzip")
                # semseg_path = info['lidar_path'].replace("pandaset_pnp_npt_cam", "pandaset").replace("lidar", "annotations/semseg")
                # semseg = pd.read_pickle(semseg_path, compression="gzip")
                # # pandar64_mask = data['d'] != 1
                # # data = data[pandar64_mask]
                # # semseg = semseg[pandar64_mask]
                # semseg_mask = semseg == 13
                # data_viz = data[np.asarray(semseg_mask).squeeze()]

                # verts = torch.tensor(np.stack([np.array(data_viz['x']), np.array(data_viz['y']), np.array(data_viz['z'])], axis=-1)).float()
                # cam_pts = transform(verts, torch.tensor(cam2world).inverse())
                # pixel_pts = transform(cam_pts, intrinsic).long().numpy()
                # # pixel_pts = transform(verts, world2screen)
                # # pixel_pts = (pixel_pts[:,:2] / pixel_pts[:,2:3]).long().numpy()

                # verts_visible_mask = (
                #     (pixel_pts[:, 0] <= 1920 - 1)
                #     & (pixel_pts[:, 1] <= 1080 - 1)
                #     & (pixel_pts[:, 0] >= 0)
                #     & (pixel_pts[:, 1] >= 0)
                #     & (cam_pts.numpy()[:, 2] >= 0)
                # )
                # proj_pts = pixel_pts[verts_visible_mask]
                # image = imageio.imread(cam_info['data_path'])
                # image_proj = np.array(image)
                # for xy in proj_pts:
                #     cv2.circle(image_proj, (xy[0], xy[1]), 1, (255, 128, 0), thickness=1)
                # import matplotlib.pyplot as plt; plt.imshow(image_proj); plt.show()
                # import pdb; pdb.set_trace()

                """
                # lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                # lidar2cam_t = cam_info[
                #     'sensor2lidar_translation'] @ lidar2cam_r.T
                # lidar2cam_rt = np.eye(4)
                # lidar2cam_rt[:3, :3] = lidar2cam_r.T
                # lidar2cam_rt[3, :3] = -lidar2cam_t
                # intrinsic = cam_info['cam_intrinsic']
                # viewpad = np.eye(4)
                # viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                # lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                # lidar2img_rts.append(lidar2img_rt)
                """

            input_dict.update(dict(img_filename=image_paths, lidar2img=lidar2img_rts, ego2world=ego2world))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # import pdb; pdb.set_trace()
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names=gt_names_3d)
        return anns_results

    # def _format_bbox(self, results, jsonfile_prefix=None):
    #     """Convert the results to the standard format.

    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         jsonfile_prefix (str): The prefix of the output jsonfile.
    #             You can specify the output directory/filename by
    #             modifying the jsonfile_prefix. Default: None.

    #     Returns:
    #         str: Path of the output json file.
    #     """
    #     nusc_annos = {}
    #     mapped_class_names = self.CLASSES

    #     print('Start to convert detection format...')
    #     for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
    #         annos = []
    #         boxes = output_to_nusc_box(det)
    #         sample_token = self.data_infos[sample_id]['token']
    #         boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
    #                                          mapped_class_names,
    #                                          self.eval_detection_configs,
    #                                          self.eval_version)
    #         for i, box in enumerate(boxes):
    #             name = mapped_class_names[box.label]
    #             if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
    #                 if name in [
    #                         'car',
    #                         'construction_vehicle',
    #                         'bus',
    #                         'truck',
    #                         'trailer',
    #                 ]:
    #                     attr = 'vehicle.moving'
    #                 elif name in ['bicycle', 'motorcycle']:
    #                     attr = 'cycle.with_rider'
    #                 else:
    #                     attr = NuScenesDataset.DefaultAttribute[name]
    #             else:
    #                 if name in ['pedestrian']:
    #                     attr = 'pedestrian.standing'
    #                 elif name in ['bus']:
    #                     attr = 'vehicle.stopped'
    #                 else:
    #                     attr = NuScenesDataset.DefaultAttribute[name]

    #             nusc_anno = dict(
    #                 sample_token=sample_token,
    #                 translation=box.center.tolist(),
    #                 size=box.wlh.tolist(),
    #                 rotation=box.orientation.elements.tolist(),
    #                 velocity=box.velocity[:2].tolist(),
    #                 detection_name=name,
    #                 detection_score=box.score,
    #                 attribute_name=attr)
    #             annos.append(nusc_anno)
    #         nusc_annos[sample_token] = annos
    #     nusc_submissions = {
    #         'meta': self.modality,
    #         'results': nusc_annos,
    #     }

    #     mmcv.mkdir_or_exist(jsonfile_prefix)
    #     res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
    #     print('Results writes to', res_path)
    #     mmcv.dump(nusc_submissions, res_path)
    #     return res_path

    # def _evaluate_single(self,
    #                      result_path,
    #                      logger=None,
    #                      metric='bbox',
    #                      result_name='pts_bbox'):
    #     """Evaluation for a single model in nuScenes protocol.

    #     Args:
    #         result_path (str): Path of the result file.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         metric (str): Metric name used for evaluation. Default: 'bbox'.
    #         result_name (str): Result name in the metric prefix.
    #             Default: 'pts_bbox'.

    #     Returns:
    #         dict: Dictionary of evaluation details.
    #     """
    #     from nuscenes import NuScenes
    #     from nuscenes.eval.detection.evaluate import NuScenesEval

    #     output_dir = osp.join(*osp.split(result_path)[:-1])
    #     nusc = NuScenes(
    #         version=self.version, dataroot=self.data_root, verbose=False)
    #     eval_set_map = {
    #         'v1.0-mini': 'mini_val',
    #         'v1.0-trainval': 'val',
    #     }
    #     nusc_eval = NuScenesEval(
    #         nusc,
    #         config=self.eval_detection_configs,
    #         result_path=result_path,
    #         eval_set=eval_set_map[self.version],
    #         output_dir=output_dir,
    #         verbose=False)
    #     nusc_eval.main(render_curves=False)

    #     # record metrics
    #     metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
    #     detail = dict()
    #     metric_prefix = f'{result_name}_NuScenes'
    #     for name in self.CLASSES:
    #         for k, v in metrics['label_aps'][name].items():
    #             val = float('{:.4f}'.format(v))
    #             detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
    #         for k, v in metrics['label_tp_errors'][name].items():
    #             val = float('{:.4f}'.format(v))
    #             detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
    #         for k, v in metrics['tp_errors'].items():
    #             val = float('{:.4f}'.format(v))
    #             detail['{}/{}'.format(metric_prefix,
    #                                   self.ErrNameMapping[k])] = val

    #     detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
    #     detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
    #     return detail

    # def format_results(self, results, jsonfile_prefix=None):
    #     """Format the results to json (standard format for COCO evaluation).

    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         jsonfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.

    #     Returns:
    #         tuple: Returns (result_files, tmp_dir), where `result_files` is a \
    #             dict containing the json filepaths, `tmp_dir` is the temporal \
    #             directory created for saving json files when \
    #             `jsonfile_prefix` is not specified.
    #     """
    #     assert isinstance(results, list), 'results must be a list'
    #     assert len(results) == len(self), (
    #         'The length of results is not equal to the dataset len: {} != {}'.
    #         format(len(results), len(self)))

    #     if jsonfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         jsonfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None

    #     # currently the output prediction results could be in two formats
    #     # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
    #     # 2. list of dict('pts_bbox' or 'img_bbox':
    #     #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
    #     # this is a workaround to enable evaluation of both formats on nuScenes
    #     # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
    #     if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
    #         result_files = self._format_bbox(results, jsonfile_prefix)
    #     else:
    #         # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
    #         result_files = dict()
    #         for name in results[0]:
    #             print(f'\nFormating bboxes of {name}')
    #             results_ = [out[name] for out in results]
    #             tmp_file_ = osp.join(jsonfile_prefix, name)
    #             result_files.update(
    #                 {name: self._format_bbox(results_, tmp_file_)})
    #     return result_files, tmp_dir

    def bbox2result_kitti(self, net_outputs, class_names, pklfile_prefix=None, submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.
        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.
        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), "invalid list length of network outputs"
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print("\nConverting prediction to KITTI format")
        for idx, pred_dicts in enumerate(mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            # sample_idx = (int(info['sequence']) - 1) * 80 + info['frame_idx']
            # image_shape = (1080, 1920)
            # box_dict = self.convert_valid_bboxes(pred_dicts, info)
            # box_dict = pred_dicts["pts_bbox"]
            box_dict = pred_dicts
            anno = {
                "name": [],
                "truncated": [],
                "occluded": [],
                # 'alpha': [],
                # 'bbox': [],
                "dimensions": [],
                "location": [],
                "rotation_y": [],
                "score": [],
                # 'boxes_3d': [],
            }
            if len(box_dict["boxes_3d"]) > 0:
                scores_3d = box_dict["scores_3d"]
                boxes_3d = box_dict["boxes_3d"]
                labels_3d = box_dict["labels_3d"]

                for box, score, label in zip(boxes_3d, scores_3d, labels_3d):
                    anno["name"].append(class_names[int(label)])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    # anno['alpha'].append(
                    #     -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    # anno['bbox'].append(bbox)
                    anno["dimensions"].append(box[3:6].numpy())
                    anno["location"].append(box[:3].numpy())
                    anno["rotation_y"].append(box[6].item())
                    # anno['boxes_3d'].append(box)
                    anno["score"].append(score.item())

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    "name": np.array([]),
                    "truncated": np.array([]),
                    "occluded": np.array([]),
                    # 'alpha': np.array([]),
                    # 'bbox': np.zeros([0, 4]),
                    "dimensions": np.zeros([0, 3]),
                    "location": np.zeros([0, 3]),
                    "rotation_y": np.array([]),
                    "score": np.array([]),
                }
                annos.append(anno)

            # if submission_prefix is not None:
            #     curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
            #     with open(curr_file, 'w') as f:
            #         bbox = anno['bbox']
            #         loc = anno['location']
            #         dims = anno['dimensions']  # lhw -> hwl

            #         for idx in range(len(bbox)):
            #             print(
            #                 '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
            #                 '{:.4f} {:.4f} {:.4f} '
            #                 '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
            #                     anno['name'][idx], anno['alpha'][idx],
            #                     bbox[idx][0], bbox[idx][1], bbox[idx][2],
            #                     bbox[idx][3], dims[idx][1], dims[idx][2],
            #                     dims[idx][0], loc[idx][0], loc[idx][1],
            #                     loc[idx][2], anno['rotation_y'][idx],
            #                     anno['score'][idx]),
            #                 file=f)

            # annos[-1]['sample_idx'] = np.array(
            #     [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith((".pkl", ".pickle")):
                out = f"{pklfile_prefix}.pkl"
            mmcv.dump(det_annos, out)
            print(f"Result is saved to {out}.")

        return det_annos

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        """Format the results to pkl file.
        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self.bbox2result_kitti(outputs, self.CLASSES, pklfile_prefix, submission_prefix)
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # self.show_gt("work_dirs/viz_debug", 0)
        # self.show(results, "/home/wangjk/viz_debug", 0)
        # import pdb; pdb.set_trace()

        # pipeline = self._build_default_pipeline()
        evaluator = Evaluator(ap_thresholds=[0.5, 1.0, 2.0, 4.0])
        for i, result in enumerate(results):
            if "pts_bbox" in result.keys():
                result = result["pts_bbox"]
            #     # data_info = self.data_infos[i]
            #     # pts_path = data_info['lidar_path']
            #     # file_name = osp.split(pts_path)[-1].split('.')[0]
            #     # points = self._extract_data(i, pipeline, 'points').numpy()
            #     # # for now we convert points into depth mode
            #     # points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
            #     #                                    Coord3DMode.DEPTH)
            #     # inds = result['scores_3d'] > 0.1
            #     # gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            #     # show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
            #     #                                    Box3DMode.DEPTH)
            #     # pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            #     # show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
            #     #                                      Box3DMode.DEPTH)

            from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter

            roi_filter = ObjectRangeFilter([0, -39.68, -1.5, 79.36, 39.68, 5.5])
            gt_label = roi_filter(self.get_ann_info(i))
            # import ipdb; ipdb.set_trace()
            # result["boxes_3d"] = gt_label["gt_bboxes_3d"]
            # result["scores_3d"] = torch.ones((result["boxes_3d"].tensor.shape[0],))
            # result["labels_3d"] = torch.zeros_like(result["scores_3d"]).long()
            evaluator.append(result, gt_label)

            # evaluator.append(result, self.get_ann_info(i))

        result = evaluator.evaluate()
        result_df = result.as_dataframe()
        res_dict = {}
        for key in result_df.keys():
            res_dict[str(key)] = result_df[key][0]
        print(res_dict)
        return res_dict

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        gt_annos = [{"gt_boxes": info["gt_boxes"], "gt_names": info["gt_names"]} for info in self.data_infos]
        ap_result_str, ap_dict = kitti_eval(gt_annos, result_files, self.CLASSES, eval_types="3d")
        print(ap_result_str)
        return ap_dict

        # if isinstance(result_files, dict):
        #     ap_dict = dict()
        #     for name, result_files_ in result_files.items():
        #         eval_types = ['bbox', 'bev', '3d']
        #         if 'img' in name:
        #             eval_types = ['bbox']
        #         ap_result_str, ap_dict_ = kitti_eval(
        #             gt_annos,
        #             result_files_,
        #             self.CLASSES,
        #             eval_types=eval_types)
        #         for ap_type, ap in ap_dict_.items():
        #             ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

        #         print(
        #             f'Results of {name}:\n')

        # else:
        #     if metric == 'img_bbox':
        #         ap_result_str, ap_dict = kitti_eval(
        #             gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
        #     else:
        #         ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
        #                                             self.CLASSES)
        #     print('\n' + ap_result_str)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()
        # if show or out_dir:
        #     self.show(results, out_dir, show=show, pipeline=pipeline)

        import pdb

        pdb.set_trace()

        # if show:
        # out_dir = "/home/wangjk/viz_debug"
        # self.show(results, out_dir, pipeline=pipeline)

        # if isinstance(result_files, dict):
        #     results_dict = dict()
        #     for name in result_names:
        #         print('Evaluating bboxes of {}'.format(name))
        #         ret_dict = self._evaluate_single(result_files[name])
        #     results_dict.update(ret_dict)
        # elif isinstance(result_files, str):
        #     results_dict = self._evaluate_single(result_files)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        # return results

        pass

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=3,
                use_dim=3,
                file_client_args=dict(backend="disk"),
            ),
            # dict(
            #     type='LoadPointsFromMultiSweeps',
            #     sweeps_num=1,
            #     file_client_args=dict(backend='disk')),
            dict(type="DefaultFormatBundle3D", class_names=self.CLASSES, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, "Expect out_dir, got none."
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if "pts_bbox" in result.keys():
                result = result["pts_bbox"]
            data_info = self.data_infos[i]
            pts_path = data_info["lidar_path"]
            file_name = osp.split(pts_path)[-1].split(".")[0]
            points = self._extract_data(i, pipeline, "points").numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)["gt_bboxes_3d"].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            pred_bboxes = result["boxes_3d"].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir, file_name, show)

    # def show(self, results, out_dir, frame_idx=0, show=True, pipeline=None):
    #     """Results visualization.

    #     Args:
    #         results (list[dict]): List of bounding boxes results.
    #         out_dir (str): Output directory of visualization result.
    #         show (bool): Visualize the results online.
    #         pipeline (list[dict], optional): raw data loading for showing.
    #             Default: None.
    #     """
    #     assert out_dir is not None, "Expect out_dir, got none."
    #     # pipeline = self._get_pipeline(pipeline)
    #     pipeline = self._build_default_pipeline()
    #     for i, result in enumerate(results):
    #         if i != frame_idx:
    #             continue
    #         if "pts_bbox" in result.keys():
    #             result = result["pts_bbox"]
    #         data_info = self.data_infos[i]
    #         pts_path = data_info["lidar_path"]
    #         file_name = osp.split(pts_path)[-1].split(".")[0]
    #         # import pdb; pdb.set_trace()
    #         points = self._extract_data(i, pipeline, "points").numpy()
    #         # for now we convert points into depth mode
    #         # points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
    #         #                                    Coord3DMode.DEPTH)
    #         from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter

    #         roi_filter = ObjectRangeFilter([-80, -40, -3, 80, 40, 1])
    #         gt_bboxes = roi_filter(self.get_ann_info(i))["gt_bboxes_3d"].tensor.numpy()
    #         # gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
    #         # show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
    #         #                                    Box3DMode.DEPTH)
    #         # pred_bboxes = roi_filter(result['boxes_3d'][inds]).tensor.numpy()
    #         inds = result["scores_3d"] > 0.3
    #         pred_bboxes = result["boxes_3d"][inds].tensor.numpy()

    #         # show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
    #         #                                      Box3DMode.DEPTH)
    #         show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name, show)

    def show_gt(self, out_dir, idx=0, show=True, pipeline=None):
        """Results visualization.

        Args:
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, "Expect out_dir, got none."
        pipeline = self._build_default_pipeline()

        data_info = self.data_infos[idx]
        pts_path = data_info["lidar_path"]
        file_name = osp.split(pts_path)[-1].split(".")[0]
        points = self._extract_data(idx, pipeline, "points").numpy()
        # points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
        #                                     Coord3DMode.DEPTH)
        from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter

        roi_filter = ObjectRangeFilter([-80, -40, -1.5, 80, 40, 5.5])

        input_dict = self.get_ann_info(idx)
        input_dict = roi_filter(input_dict)

        gt_bboxes = input_dict["gt_bboxes_3d"].tensor.numpy()
        # gt_bboxes = self.get_ann_info(idx)['gt_bboxes_3d'].tensor.numpy()
        # show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
        #                                     Box3DMode.DEPTH)
        show_result(points, gt_bboxes, gt_bboxes, out_dir, file_name, show)


# def output_to_nusc_box(detection):
#     """Convert the output to the box class in the nuScenes.

#     Args:
#         detection (dict): Detection results.

#             - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
#             - scores_3d (torch.Tensor): Detection scores.
#             - labels_3d (torch.Tensor): Predicted box labels.

#     Returns:
#         list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
#     """
#     box3d = detection['boxes_3d']
#     scores = detection['scores_3d'].numpy()
#     labels = detection['labels_3d'].numpy()

#     box_gravity_center = box3d.gravity_center.numpy()
#     box_dims = box3d.dims.numpy()
#     box_yaw = box3d.yaw.numpy()
#     # TODO: check whether this is necessary
#     # with dir_offset & dir_limit in the head
#     box_yaw = -box_yaw - np.pi / 2

#     box_list = []
#     for i in range(len(box3d)):
#         quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
#         velocity = (*box3d.tensor[i, 7:9], 0.0)
#         # velo_val = np.linalg.norm(box3d[i, 7:9])
#         # velo_ori = box3d[i, 6]
#         # velocity = (
#         # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
#         box = NuScenesBox(
#             box_gravity_center[i],
#             box_dims[i],
#             quat,
#             label=labels[i],
#             score=scores[i],
#             velocity=velocity)
#         box_list.append(box)
#     return box_list


# def lidar_nusc_box_to_global(info,
#                              boxes,
#                              classes,
#                              eval_configs,
#                              eval_version='detection_cvpr_2019'):
#     """Convert the box from ego to global coordinate.

#     Args:
#         info (dict): Info for a specific sample data, including the
#             calibration information.
#         boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
#         classes (list[str]): Mapped classes in the evaluation.
#         eval_configs (object): Evaluation configuration object.
#         eval_version (str): Evaluation version.
#             Default: 'detection_cvpr_2019'

#     Returns:
#         list: List of standard NuScenesBoxes in the global
#             coordinate.
#     """
#     box_list = []
#     for box in boxes:
#         # Move box to ego vehicle coord system
#         box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
#         box.translate(np.array(info['lidar2ego_translation']))
#         # filter det in ego.
#         cls_range_map = eval_configs.class_range
#         radius = np.linalg.norm(box.center[:2], 2)
#         det_range = cls_range_map[classes[box.label]]
#         if radius > det_range:
#             continue
#         # Move box to global coord system
#         box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
#         box.translate(np.array(info['ego2global_translation']))
#         box_list.append(box)
#     return box_list


from dataclasses import dataclass
from typing import List, Dict

import torch

from dataclasses import dataclass


@dataclass
class EvaluationFrame:
    """Dataclass to store the evaluation inputs for one frame."""

    detections: Dict
    labels: Dict


@dataclass
class Matching:
    """Matching between N detections and M ground truth labels.
    Attributes:
        scores: [N] detection scores, sorted in decreasing order.
        true_positives: [N] booleans indicating whether a detection is a true positive.
        false_negatives: [M] booleans indicating whether a label is a false negative.
    """

    scores: torch.Tensor
    true_positives: torch.Tensor
    false_negatives: torch.Tensor

    def __post_init__(self) -> None:
        """Assert data structure invariants."""
        assert (
            self.scores.shape == self.true_positives.shape
        ), "`scores` and `true_positives` should have the same shape."
        assert self.scores.ndim == 1, "`scores` should have one dimension only."
        assert self.true_positives.ndim == 1, "`true_positives` should have one dimension only."
        assert self.false_negatives.ndim == 1, "`false_negatives` should have one dimension only."
        assert (
            self.scores.device == self.true_positives.device == self.false_negatives.device
        ), "`scores`, `true_positives`, and `false_negatives` must be on the same device."
        assert self.true_positives.dtype == torch.bool, "`true_positives.dtype` should equal `torch.bool`."
        assert self.false_negatives.dtype == torch.bool, "`true_negatives.dtype` should equal `torch.bool`."

    @property
    def device(self) -> torch.device:
        """The device of this data structure's tensors."""
        return self.true_positives.device


def compute_matching_by_distance(
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    target_bboxes: torch.Tensor,
    distance_threshold: float,
) -> Matching:
    """Return a matching between N detections and M ground truth labels.
    Args:
        bboxes: [N x 2] detected BEV bounding box centroids (x, y).
        scores: [N] detection confidence scores.
        target_bboxes: [M x 2] target BEV bounding box centroids (x, y).
        distance_threshold: Two bounding boxes match if their centroid-to-centroid
            distance is strictly less than `distance_threshold`.
    """
    true_positives = torch.zeros(len(bboxes), dtype=torch.bool, device=bboxes.device)
    if len(target_bboxes) == 0:
        false_negatives = torch.zeros_like(target_bboxes[:, 0], dtype=torch.bool)
        return Matching(scores.clone(), true_positives, false_negatives)

    distance_matrix = torch.norm(bboxes[:, None, :3] - target_bboxes[None, :, :3], dim=-1)
    min_distances, min_distance_indices = torch.min(distance_matrix, dim=1)

    matched = torch.zeros(len(target_bboxes), dtype=torch.bool, device=bboxes.device)
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    for bbox_index in sorted_indices:
        if min_distances[bbox_index] < distance_threshold:
            if not matched[min_distance_indices[bbox_index]]:
                true_positives[bbox_index] = True
                matched[min_distance_indices[bbox_index]] = True

    return Matching(scores.clone(), true_positives, ~matched)


def merge_matchings(matchings: List[Matching]) -> Matching:
    """Merge a list of matchings."""
    if len(matchings) == 0:
        return Matching(
            torch.zeros(0, dtype=torch.float),
            torch.zeros(0, dtype=torch.bool),
            torch.zeros(0, dtype=torch.bool),
        )

    scores = torch.cat([m.scores for m in matchings], dim=0)
    true_positives = torch.cat([m.true_positives for m in matchings], dim=0)
    false_negatives = torch.cat([m.false_negatives for m in matchings], dim=0)
    return Matching(scores, true_positives, false_negatives)


@dataclass
class PRCurve:
    """A precision/recall curve.
    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor

    def __post_init__(self) -> None:
        """Assert data structure invariants."""
        assert self.precision.ndim == 1, "`precision` should have one dimension only."
        assert self.recall.ndim == 1, "`recall` should have one dimension only."
        assert self.precision.shape == self.recall.shape, "`precision` and `recall` should have the same shape."
        assert self.precision.dtype == self.recall.dtype, "`precision` and `recall` should have the same dtype."
        assert self.precision.device == self.recall.device, "`precision` and `recall` should be on the same device."


def compute_precision_recall_curve_from_matching(matching: Matching) -> PRCurve:
    """Return the precision-recall curve from the given matching."""
    num_true_positives = torch.sum(matching.true_positives)
    num_false_negatives = torch.sum(matching.false_negatives)
    num_labels = num_true_positives + num_false_negatives
    if num_labels == 0:
        return PRCurve(
            torch.zeros_like(matching.scores),
            torch.zeros_like(matching.scores),
        )

    sorted_indices = torch.argsort(matching.scores, dim=0, descending=True)
    cumulative_true_positives = torch.cumsum(matching.true_positives[sorted_indices], dim=0)
    cumulative_num_detections = torch.arange(1, len(cumulative_true_positives) + 1, device=matching.device)

    precision = cumulative_true_positives / cumulative_num_detections
    recall = cumulative_true_positives / num_labels
    return PRCurve(
        precision.to(matching.scores.dtype),
        recall.to(matching.scores.dtype),
    )


def compute_precision_recall_curve(frames: List[EvaluationFrame], threshold: float) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.
    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.
    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.
    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.
    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.
    Returns:
        A precision/recall curve.
    """
    # TODO: Implement.
    matchings = []
    for frame in frames:
        matchings.append(
            compute_matching_by_distance(
                frame.detections["boxes_3d"].tensor[:, :3],
                frame.detections["scores_3d"],
                frame.labels["gt_bboxes_3d"].tensor[:, :3],
                threshold,
            )
        )

    matching = merge_matchings(matchings)
    return compute_precision_recall_curve_from_matching(matching)


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.
    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.
    Args:
        curve: The precision/recall curve.
    Returns:
        The area under the curve, as defined above.
    """
    # TODO: Implement.
    dx = curve.recall - torch.cat([torch.zeros_like(curve.recall[:1]), curve.recall[:-1]], dim=0)
    return torch.sum(curve.precision * dx).item()


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_average_precision(frames: List[EvaluationFrame], threshold: float) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.
    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.
    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # TODO: Implement.
    pr_curve = compute_precision_recall_curve(frames, threshold)
    ap = compute_area_under_curve(pr_curve)
    return AveragePrecisionMetric(ap, pr_curve)


from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class EvaluationResult:
    """Dataclass to store the results of an evaluation."""

    def __init__(self, ap_metrics: Dict[float, AveragePrecisionMetric]) -> None:
        self._ap_metrics = ap_metrics

    @property
    def mean_ap(self) -> float:
        """Return the mean average precision over all thresholds."""
        if len(self._ap_metrics) == 0:
            return 0.0
        return np.mean(np.array([m.ap for m in self._ap_metrics.values()]))

    @property
    def ap_metrics(self) -> Dict[float, AveragePrecisionMetric]:
        """Return average precision metrics broken down by threshold."""
        return self._ap_metrics

    def as_dataframe(self) -> pd.DataFrame:
        """Return average precision as a data frame."""
        ap_dict = {th: [m.ap] for th, m in self._ap_metrics.items()}
        ap_dict["mean"] = [self.mean_ap]
        return pd.DataFrame.from_dict(ap_dict)

    def visualize(self, figsize: Tuple[int, int] = (8, 16), dpi: int = 75) -> Optional[Tuple[Figure, Axes]]:
        """Visualize the evaluation results in matplotlib.
        Returns:
            Matplotlib figure and axis. `fig.show()` will display result.
        """
        if len(self._ap_metrics) == 0:
            return None

        fig, axes = plt.subplots(len(self._ap_metrics), figsize=figsize, dpi=dpi)
        for index, threshold in enumerate(self._ap_metrics.keys()):
            metric = self._ap_metrics[threshold]
            axes[index].plot(
                metric.pr_curve.recall.cpu().numpy(),
                metric.pr_curve.precision.cpu().numpy(),
            )
            axes[index].set_title(f"AP = {metric.ap:.2f} (threshold = {threshold:.1f}m)")
        fig.supxlabel("Recall")
        fig.supylabel("Precision")
        fig.suptitle("Precision-Recall Curves")
        return fig, axes


class Evaluator:
    """Evaluates detections against ground truth labels."""

    def __init__(self, ap_thresholds: List[float]) -> None:
        """Initialization.
        Args:
            ap_thresholds: The thresholds used to evaluate AP.
        """
        self._ap_thresholds = ap_thresholds
        self._evaluation_frames: List[EvaluationFrame] = []

    def append(self, detections: Dict, labels: Dict) -> None:
        """Buffer a frame of detections and labels into the evaluator."""
        self._evaluation_frames.append(
            EvaluationFrame(
                detections,
                labels,
            )
        )

    def evaluate(self) -> EvaluationResult:
        """Evaluate the buffered frames and return results."""
        ap_metrics = {}
        for threshold in self._ap_thresholds:
            metric = compute_average_precision(self._evaluation_frames, threshold)
            ap_metrics[threshold] = metric
        return EvaluationResult(ap_metrics)

    def reset(self) -> None:
        """Reset the buffer."""
        self._evaluation_frames = []

    def __len__(self) -> int:
        """Return the size of the buffer."""
        return len(self._evaluation_frames)


##############################################################3

# Copyright (c) OpenMMLab. All rights reserved.
import gc
import io as sysio
import numba
import numpy as np


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1)):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ["car", "pedestrian", "cyclist"]
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["gt_names"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["gt_boxes"][i]
        gt_name = gt_anno["gt_names"][i].lower()
        # height = bbox[3] - bbox[1]
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == "Pedestrian".lower() and "Person_sitting".lower() == gt_name:
            valid_class = 0
        elif current_cls_name == "Car".lower() and "Van".lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if False:
            # if ((gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty])
            #         or (gt_anno['truncated'][i] > MAX_TRUNCATION[difficulty])
            #         # or (height <= MIN_HEIGHT[difficulty])
            #     ):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
    # if gt_anno['gt_names'][i] == 'DontCare':
    #     dc_bboxes.append(gt_anno['bbox'][i])
    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        # height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
        if False:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (query_boxes[k, 3] - query_boxes[k, 1])
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0])
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1])
                if ih > 0:
                    if criterion == -1:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih
                    elif criterion == 0:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval

    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = min(boxes[i, 1], qboxes[j, 1]) - max(boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4])

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = area1 + area2 - inc
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    from mmdet3d.core.evaluation.kitti_utils.rotate_iou import rotate_iou_gpu_eval

    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    dc_bboxes,
    metric,
    min_overlap,
    thresh=0,
    compute_fp=False,
    compute_aos=False,
):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    # dt_alphas = dt_datas[:, 4]
    # gt_alphas = gt_datas[:, 4]
    # dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if not compute_fp and (overlap > min_overlap) and dt_score > valid_detection:
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif compute_fp and (overlap > min_overlap) and (valid_detection == NO_DETECTION) and ignored_det[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            # if compute_aos:
            #     # delta.append(gt_alphas[i] - dt_alphas[det_idx])
            #     delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
            #     delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (assigned_detection[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i]):
                fp += 1
        nstuff = 0
        # if metric == 0:
        #     overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
        #     for i in range(dc_bboxes.shape[0]):
        #         for j in range(det_size):
        #             if (assigned_detection[j]):
        #                 continue
        #             if (ignored_det[j] == -1 or ignored_det[j] == 1):
        #                 continue
        #             if (ignored_threshold[j]):
        #                 continue
        #             if overlaps_dt_dc[j, i] > min_overlap:
        #                 assigned_detection[j] = True
        #                 nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    dc_nums,
    gt_datas,
    dt_datas,
    dontcares,
    ignored_gts,
    ignored_dets,
    metric,
    min_overlap,
    thresholds,
    compute_aos=False,
):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num : dt_num + dt_nums[i], gt_num : gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num : gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num : dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num : gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num : dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num : dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos,
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["gt_names"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            # loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
            # dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            # rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            # gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
            #                           axis=1)
            gt_boxes = np.concatenate([a["gt_boxes"] for a in gt_annos_part], 0)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx : gt_num_idx + gt_box_num, dt_num_idx : dt_num_idx + dt_box_num]
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = gt_annos[i]["gt_boxes"]  # np.concatenate(
        #     [gt_annos[i]['gt_boxes'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        dt_datas = np.concatenate(
            [dt_annos[i]["location"], dt_annos[i]["dimensions"], dt_annos[i]["rotation_y"][..., np.newaxis]], axis=1
        )
        # dt_datas = np.concatenate([
        #     dt_annos[i]['bbox'], # dt_annos[i]['alpha'][..., np.newaxis],
        #     dt_annos[i]['score'][..., np.newaxis]
        # ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt)


def eval_class(
    gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps, compute_aos=False, num_parts=200
):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        current_classes (list[int]): 0: car, 1: pedestrian, 2: cyclist.
        difficultys (list[int]): Eval difficulty, 0: easy, 1: normal, 2: hard
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps (float): Min overlap. format:
            [num_overlap, metric, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(gt_annos, dt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for idx_l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                total_dc_num,
                total_num_valid_gt,
            ) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False,
                    )
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(gt_datas_list[idx : idx + num_part], 0)
                    dt_datas_part = np.concatenate(dt_datas_list[idx : idx + num_part], 0)
                    dc_datas_part = np.concatenate(dontcares[idx : idx + num_part], 0)
                    ignored_dets_part = np.concatenate(ignored_dets[idx : idx + num_part], 0)
                    ignored_gts_part = np.concatenate(ignored_gts[idx : idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx : idx + num_part],
                        total_dt_num[idx : idx + num_part],
                        total_dc_num[idx : idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos,
                    )
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, idx_l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, idx_l, k, i] = np.max(precision[m, idx_l, k, i:], axis=-1)
                    recall[m, idx_l, k, i] = np.max(recall[m, idx_l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, idx_l, k, i] = np.max(aos[m, idx_l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }

    # clean temp variables
    del overlaps
    del parted_overlaps

    gc.collect()
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos, dt_annos, current_classes, min_overlaps, eval_types=["bbox", "bev", "3d"]):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    mAP_bbox = None
    mAP_aos = None
    if "bbox" in eval_types:
        ret = eval_class(
            gt_annos, dt_annos, current_classes, difficultys, 0, min_overlaps, compute_aos=("aos" in eval_types)
        )
        # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
        mAP_bbox = get_mAP(ret["precision"])
        if "aos" in eval_types:
            mAP_aos = get_mAP(ret["orientation"])

    mAP_bev = None
    if "bev" in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1, min_overlaps)
        mAP_bev = get_mAP(ret["precision"])

    mAP_3d = None
    if "3d" in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2, min_overlaps)
        mAP_3d = get_mAP(ret["precision"])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def kitti_eval(gt_annos, dt_annos, current_classes, eval_types=["bbox", "bev", "3d"]):
    """KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, "must contain at least one evaluation type"
    if "aos" in eval_types:
        assert "bbox" in eval_types, "must evaluate bbox when evaluating aos"
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5], [0.7, 0.5, 0.5, 0.7, 0.5], [0.7, 0.5, 0.5, 0.7, 0.5]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25], [0.5, 0.25, 0.25, 0.5, 0.25]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: "Car",
        1: "Pedestrian",
        2: "Cyclist",
        # 3: 'Van',
        # 4: 'Person_sitting',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ""
    # check whether alpha is valid
    compute_aos = False
    # pred_alpha = False
    # valid_alpha_gt = False
    # for anno in dt_annos:
    #     mask = (anno['alpha'] != -10)
    #     if anno['alpha'][mask].shape[0] != 0:
    #         pred_alpha = True
    #         break
    # for anno in gt_annos:
    #     if anno['alpha'][0] != -10:
    #         valid_alpha_gt = True
    #         break
    # compute_aos = (pred_alpha and valid_alpha_gt)
    if compute_aos:
        eval_types.append("aos")

    mAPbbox, mAPbev, mAP3d, mAPaos = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, eval_types)

    ret_dict = {}
    difficulty = ["easy", "moderate", "hard"]
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += "{} AP@{:.2f}, {:.2f}, {:.2f}:\n".format(curcls_name, *min_overlaps[i, :, j])
            if mAPbbox is not None:
                result += "bbox AP:{:.4f}, {:.4f}, {:.4f}\n".format(*mAPbbox[j, :, i])
            if mAPbev is not None:
                result += "bev  AP:{:.4f}, {:.4f}, {:.4f}\n".format(*mAPbev[j, :, i])
            if mAP3d is not None:
                result += "3d   AP:{:.4f}, {:.4f}, {:.4f}\n".format(*mAP3d[j, :, i])

            if compute_aos:
                result += "aos  AP:{:.2f}, {:.2f}, {:.2f}\n".format(*mAPaos[j, :, i])

            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f"{difficulty[idx]}_strict"
                else:
                    postfix = f"{difficulty[idx]}_loose"
                prefix = f"KITTI/{curcls_name}"
                if mAP3d is not None:
                    ret_dict[f"{prefix}_3D_{postfix}"] = mAP3d[j, idx, i]
                if mAPbev is not None:
                    ret_dict[f"{prefix}_BEV_{postfix}"] = mAPbev[j, idx, i]
                if mAPbbox is not None:
                    ret_dict[f"{prefix}_2D_{postfix}"] = mAPbbox[j, idx, i]

    # calculate mAP over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += "\nOverall AP@{}, {}, {}:\n".format(*difficulty)
        if mAPbbox is not None:
            mAPbbox = mAPbbox.mean(axis=0)
            result += "bbox AP:{:.4f}, {:.4f}, {:.4f}\n".format(*mAPbbox[:, 0])
        if mAPbev is not None:
            mAPbev = mAPbev.mean(axis=0)
            result += "bev  AP:{:.4f}, {:.4f}, {:.4f}\n".format(*mAPbev[:, 0])
        if mAP3d is not None:
            mAP3d = mAP3d.mean(axis=0)
            result += "3d   AP:{:.4f}, {:.4f}, {:.4f}\n".format(*mAP3d[:, 0])
        if compute_aos:
            mAPaos = mAPaos.mean(axis=0)
            result += "aos  AP:{:.2f}, {:.2f}, {:.2f}\n".format(*mAPaos[:, 0])

        # prepare results for logger
        for idx in range(3):
            postfix = f"{difficulty[idx]}"
            if mAP3d is not None:
                ret_dict[f"KITTI/Overall_3D_{postfix}"] = mAP3d[idx, 0]
            if mAPbev is not None:
                ret_dict[f"KITTI/Overall_BEV_{postfix}"] = mAPbev[idx, 0]
            if mAPbbox is not None:
                ret_dict[f"KITTI/Overall_2D_{postfix}"] = mAPbbox[idx, 0]

    return result, ret_dict


def kitti_eval_coco_style(gt_annos, dt_annos, current_classes):
    """coco style evaluation of kitti.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.

    Returns:
        string: Evaluation results.
    """
    class_to_name = {
        0: "Car",
        1: "Pedestrian",
        2: "Cyclist",
        3: "Van",
        4: "Person_sitting",
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(class_to_range[curcls])[:, np.newaxis]
    result = ""
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno["alpha"].shape[0] != 0:
            if anno["alpha"][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos
    )
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} " "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, " f"{mAPbbox[j, 1]:.2f}, " f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, " f"{mAPbev[j, 1]:.2f}, " f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, " f"{mAP3d[j, 1]:.2f}, " f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, " f"{mAPaos[j, 1]:.2f}, " f"{mAPaos[j, 2]:.2f}"))
    return result
