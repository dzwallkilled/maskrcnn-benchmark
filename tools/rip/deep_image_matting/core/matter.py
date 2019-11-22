import torch
import argparse
import cv2
import os
import numpy as np
from PIL import Image
from skimage import measure
import re
from itertools import groupby
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from .deploy import inference_img_whole
from .net import VGG16

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)

    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


class Matter:
    def __init__(self, cuda=True, resume='', stage=1, crop_or_resize='whole', max_size=1920):
        self.args = argparse.ArgumentParser().parse_args()
        self.args.cuda = cuda
        self.args.resume = resume
        self.args.stage = stage
        self.args.crop_or_resize = crop_or_resize
        self.args.max_size = max_size
        model = VGG16(self.args)
        ckpt = torch.load(self.args.resume)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        if cuda:
            model = model.cuda()
        self.model = model

    def matting(self, image, trimap):
        if trimap.dtype is not np.uint8:
            assert trimap.dtype == np.float or trimap.dtype == np.float32
            trimap = (trimap * 255).astype(np.uint8)
        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_mattes = inference_img_whole(self.args, self.model, image, trimap)
        pred_mattes = (pred_mattes * 255).astype(np.uint8)
        pred_mattes[trimap == 255] = 255
        pred_mattes[trimap == 0] = 0
        return pred_mattes


if __name__ == "__main__":
    from generate_masks import MaskSaver
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    root = os.path.expanduser('~/data/RipData/RipTrainingAllData')
    anno_file = os.path.expanduser('~/data/RipData/COCOJSONs/full/rip_data_train.json')
    dataset = MaskSaver(root, anno_file)
    coco_example = COCO(anno_file)
    dataset.step_gif('img_cv.png', 'mask.png', index=0)

    result_dir = './tests'
    os.makedirs(result_dir, exist_ok=True)

    # parameters setting
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cuda = True
    args.resume = "../model/stage1_sad_54.4.pth"
    args.stage = 1
    args.crop_or_resize = "whole"
    args.max_size = 1600

    # init model
    model = VGG16(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda()

    # infer one by one
    # for image_path, trimap_path in zip(image_list, trimap_list):
    for idx, (image, trimap) in enumerate(dataset):
        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_mattes = inference_img_whole(args, model, image, trimap)

        pred_mattes = (pred_mattes * 255).astype(np.uint8)
        pred_mattes[trimap == 255] = 255
        pred_mattes[trimap == 0] = 0

        cv2.imwrite(os.path.join(result_dir, 'res_img_cv.png'), pred_mattes)

        mask_com = (trimap.astype(np.float32) * pred_mattes.astype(np.float32))
        mask_com = (mask_com / np.max(mask_com) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(result_dir, 'mask_com.png'), mask_com)

        img_com = 0.8 * image + 0.2 * np.repeat(mask_com, 3).reshape((1080, 1920, 3))
        cv2.imwrite(os.path.join(result_dir, 'com_img_cv.png'), img_com)

        guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dst1 = cv2.ximgproc.guidedFilter(
            guide=guide, src=pred_mattes, radius=4, eps=50, dDepth=-1)
        cv2.imwrite(os.path.join(result_dir, 'res_dst.png'), dst1)
        img_dst = 0.8 * image + 0.2 * np.repeat(dst1, 3).reshape((1080, 1920, 3))
        cv2.imwrite(os.path.join(result_dir, 'com_img_dst.png'), img_dst)


        # mask = pred_mattes
        mask = pred_mattes > 255//10
        # mask = np.zeros((1080, 1920), dtype=np.uint8)
        # mask[500:1000, 400:800] = 1

        # res = binary_mask_to_polygon(pred_mattes*1./255, tolerance=2)
        # res1 = binary_mask_to_polygon(pred_mattes, tolerance=2)
        # res2 = binary_mask_to_rle(pred_mattes*1./255)

        res = binary_mask_to_polygon(mask, tolerance=2)
        import time
        tic = time.time()
        for _i in range(100):
            res2 = binary_mask_to_rle(mask)
        toc = time.time()
        print(toc - tic)

        tic = time.time()
        for _i in range(100):
            pred_mattes[pred_mattes > 10] = 255
            pred_mattes[pred_mattes <= 10] = 0
            rle2 = maskUtils.encode(np.asfortranarray(pred_mattes))
        toc = time.time()
        print(toc - tic)

        info = dataset.get_info(idx)
        ann = {
            "id": 1,
            "image_id": 1,
            "iscrowd": 0,
            "bbox": [],
            "segmentation": res2,
            "width": 1920,
            "height": 1080,
        }
        rle = maskUtils.frPyObjects([ann['segmentation']], ann['height'], ann['width'])
        m = maskUtils.decode(rle)

        pred_mattes[pred_mattes > 25] = 255
        pred_mattes[pred_mattes <= 25] = 0
        rle2 = maskUtils.encode(np.asfortranarray(pred_mattes))
        m2 = maskUtils.decode(rle2)

        import cv2
        cv2.imwrite('./tests/mask.png', pred_mattes)
        cv2.imwrite('./tests/mask2.png', m2*255)
        cv2.imwrite('./tests/mask3.png', m*255)
        pass
        exit()

