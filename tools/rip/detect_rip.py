# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
import json

from maskrcnn_benchmark.config import cfg
# from rip_detector import RIPDetector
from rip_detector_patches import RIPDetector
from image_decomposition import decompose_image
import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument('--root', type=str, default='/data2/data2/zewei/data/RipData/RipTrainingAllData')
    parser.add_argument('--anno_file', type=str, default='/data2/data2/zewei/data/RipData/rip_data_all.json')
    parser.add_argument('--output_dir', type=str, default='/data2/data2/zewei/data/RipData/MaskRCNN')
    parser.add_argument(
        "--config-file",
        default="../../configs/rip_mask_patches/e2e_mask_rcnn_R_50_FPN_1x_level2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=1,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # args.show_mask_heatmaps = False
    # prepare object that handles inference plus adds predictions on top of image
    rip_demo = RIPDetector(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    # args.root = '/home/zd027/RipData_Debug'
    # args.anno_file = ''
    #
    # if args.anno_file.endswith('.json'):
    #     ann_file = json.load(open(args.anno_file, 'r'))
    #     files = [_img['file_name'] for _img in ann_file['images']]
    #     save_files = [os.path.join(args.output_dir, _file).replace('img', 'mask_rcnn') for _file in files]
    # else:
    #     files = os.listdir(args.root)
    #     os.makedirs(f'tests/detect/', exist_ok=True)
    #     save_files = [f'tests/detect/{_file}' for _file in files]

    args.root = '.'
    files = ['tests/img_cv.png']
    if args.show_mask_heatmaps:
        save_files = [f'tests/seg_mask/img_mask.png']
    else:
        save_files = [f'tests/detect_mask/img_mask.png']

    # for _idx, (_file, _save_file) in enumerate(zip(files, save_files)):
    #     start_time = time.time()
    #     img = cv2.imread(os.path.join(args.root, _file))
    #     patches = decompose_image(img, None, (800, 800), (300, 700))
    #     for i, patch in patches.items():
    #         composite = rip_demo.run_on_opencv_image(patch.image, 3)
    #         print("{} {} Time: {:.2f} s / img".format(_idx, _file, time.time() - start_time))
    #         frame, ext = _save_file.split('.')
    #         cv2.imwrite(frame+"_patch_%02d."%i+ext, composite)

    for _idx, (_file, _save_file) in enumerate(zip(files, save_files)):
        start_time = time.time()
        img = cv2.imread(os.path.join(args.root, _file))
        composite = rip_demo.run_on_image_patches(img, 3)
        print("{} {} Time: {:.2f} s / img".format(_idx, _file, time.time() - start_time))
        cv2.imwrite(_save_file, composite)


if __name__ == "__main__":
    main()
