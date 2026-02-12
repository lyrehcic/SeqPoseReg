# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import json
import numpy as np
from argparse import ArgumentParser
import cv2
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    all_pose_results = []

    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            break

        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            img,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # Convert numpy arrays to lists for JSON serialization
        pose_results_serializable = []
        for result in pose_results:
            result_serializable = {
                'keypoints': result['keypoints'].tolist(),  # Convert keypoints to a list
                'score': result['score'].tolist(),  # Convert keypoints scores to a list
                # Add other necessary fields here
                'area': result['area'].tolist(),
            }
            pose_results_serializable.append(result_serializable)

        all_pose_results.append(pose_results_serializable)

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)
        
        height, width = img.shape[:2]

        # 创建一个与 img 同样尺寸的白色图片
        #white_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 使用白色图片替代 img
        #vis_img = vis_pose_result(
        #    pose_model,
        #    white_img,  # 替换 img 为 white_img
        #    pose_results,
        #    radius=args.radius,
        #    thickness=args.thickness,
        #    dataset=dataset,
        #    dataset_info=dataset_info,
        #    kpt_score_thr=args.kpt_thr,
        #    show=False
        #)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Writing all pose results to a JSON file
    output_dir = 'Penn_experiment/baseball'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, '2026023.json'), 'w') as f:
        json.dump(all_pose_results, f)

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
