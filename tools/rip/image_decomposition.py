"""
This file provides codes of image decomposition and assemble
An image is decomposed into overlapping patches based on sliding cutting
The image could be re-assembled from those patches.

without specific notation, bbox denotes [x, y, w, h]
bbox_xyxy denotes [x1, y1, x2, y2]
where x=x1, y=y1, w=x2-x1, h=y2-y1
and maybe cx=(x1+x2)/2, cy=(y1+y2)/2
"""
import math
import numpy as np
import cv2
import os
import shutil
from collections import defaultdict, namedtuple
import json
import tqdm

import copy
import torch
from pycocotools import mask as maskUtils
from convert_rip_to_coco import load_folder_files

Patch = namedtuple('Patch', ['image', 'annos'])


def pad_image(input, pad, mode='constant', value=0):
    """

    :param input: input image
    :param pad: list, [top, bottom, left, right]
    :param mode: currently 'constant' only
    :param value:
    :return: image
    """
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= len(input.shape), 'Padding length too large'

    if mode == 'constant':
        output = cv2.copyMakeBorder(input, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=value)
    else:
        raise NotImplementedError()

    return output


def compute_overlapping_box(bbox1, bbox2, type='xyxy'):
    maxx1, maxy1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    if type == 'xyxy':
        minx2, miny2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        if maxx1 >= minx2 or maxy1 >= miny2:
            return []
        else:
            return [maxx1, maxy1, minx2, miny2]
    elif type == 'xywh':
        minx2, miny2 = min(bbox1[2] + bbox1[0], bbox2[2] + bbox2[0]), min(bbox1[3] + bbox1[1], bbox2[3] + bbox2[1])
        if maxx1 >= minx2 or maxy1 >= miny2:
            return []
        else:
            return [maxx1, maxy1, minx2 - maxx1, miny2 - maxy1]
    else:
        raise NotImplementedError(f'{type} not recognized.')


def decompose_image(image, annos, patch_size, stride):
    """

    :param image: input image
    :param annos:
    :param patch_size: size of patches
    :param stride: overlapping ratio between patches
    :return: a list of image patches
    """
    img_h, img_w, img_d = image.shape

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    else:
        assert isinstance(patch_size, list) or \
               isinstance(patch_size, tuple) or \
               isinstance(patch_size, np.dtype)
    if isinstance(stride, int):
        stride = (stride, stride)
    else:
        assert isinstance(stride, list) or \
               isinstance(stride, tuple) or \
               isinstance(stride, np.dtype)

    num_rows = math.ceil((img_h - patch_size[0]) / stride[0]) + 1
    num_cols = math.ceil((img_w - patch_size[1]) / stride[1]) + 1

    pad_h = (num_rows - 1) * stride[0] + patch_size[0] - img_h
    pad_w = (num_cols - 1) * stride[1] + patch_size[1] - img_w

    image = pad_image(image, [0, pad_h, 0, pad_w])

    patches = defaultdict(Patch)
    patch_cnt = 0
    for i in range(1, num_rows + 1):
        for j in range(1, num_cols + 1):
            up_left_x = stride[1] * (j - 1)
            up_left_y = stride[0] * (i - 1)
            down_right_x = up_left_x + patch_size[1] - 1
            down_right_y = up_left_y + patch_size[0] - 1
            cut_patch = image[up_left_y:down_right_y + 1, up_left_x:down_right_x + 1]

            if annos is not None:
                objects = []
                for _obj in annos['objects']:
                    new_obj = copy.deepcopy(_obj)
                    exterior = new_obj['points']['exterior']
                    bbox_xyxy = [exterior[0][0],
                            exterior[0][1],
                            exterior[1][0],
                            exterior[1][1]]
                    bbox_xyxy = [min(bbox_xyxy[0], bbox_xyxy[2]), min(bbox_xyxy[1], bbox_xyxy[3]), max(bbox_xyxy[0], bbox_xyxy[2]), max(bbox_xyxy[1], bbox_xyxy[3])]
                    bbox1 = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]]
                    assert bbox1 == _obj['bbox'], (bbox1, _obj['bbox'])
                    overlapping_bbox = compute_overlapping_box([up_left_x, up_left_y, down_right_x, down_right_y],
                                                               bbox_xyxy)

                    if overlapping_bbox:
                        new_exterior = [[overlapping_bbox[0] - up_left_x, overlapping_bbox[1] - up_left_y],
                                        [overlapping_bbox[2] - up_left_x, overlapping_bbox[3] - up_left_y]]
                        new_obj['points']['exterior'] = new_exterior
                        new_obj['bbox'] = [new_exterior[0][0],
                                           new_exterior[0][1],
                                           new_exterior[1][0] - new_exterior[0][0],
                                           new_exterior[1][1] - new_exterior[0][1]]

                        segmentation = new_obj['mask']
                        m = maskUtils.decode(segmentation)
                        m = pad_image(m, [0, pad_h, 0, pad_w])
                        m_patch = m[up_left_y:down_right_y + 1, up_left_x:down_right_x + 1]
                        seg_patch = maskUtils.encode(np.asfortranarray(m_patch))
                        seg_patch['counts'] = seg_patch['counts'].decode('utf-8')
                        new_obj['mask'] = seg_patch

                        objects.append(new_obj)
                patch_annos = {'description': annos['description'],
                               'tags': annos['tags'],
                               'size': {'height': patch_size[0], 'width': patch_size[1]},
                               'objects': objects}
                patches[patch_cnt] = Patch(image=cut_patch.copy(), annos=patch_annos.copy())
            else:
                patches[patch_cnt] = Patch(image=cut_patch.copy(), annos=None)
            patch_cnt += 1

    return patches


def assemble_patches(bboxes, patch_size, stride, image_size=(1080, 1920), type='xyxy'):
    new_bboxes = []
    # num_rows = math.ceil((image_size[0] - patch_size[0]) / stride[0]) + 1
    num_cols = math.ceil((image_size[1] - patch_size[1]) / stride[1]) + 1
    if type == 'xyxy':
        for patch_cnt, patch_bboxes in enumerate(bboxes):
            top_left_x = (patch_cnt % num_cols) * stride[1]
            top_left_y = (patch_cnt // num_cols) * stride[0]
            new_bboxes.append(
                [bbox[0] + top_left_x, bbox[1] + top_left_y, bbox[2] + top_left_x, bbox[3] + top_left_y] for bbox in
                patch_bboxes)

    elif type == 'xywh':
        for patch_cnt, patch_bboxes in enumerate(bboxes):
            top_left_x = (patch_cnt % num_cols) * stride[1]
            top_left_y = (patch_cnt // num_cols) * stride[0]
            new_bboxes.append([bbox[0] + top_left_x, bbox[1] + top_left_y, bbox[2], bbox[3]] for bbox in patch_bboxes)
    else:
        raise NotImplementedError(f'{type} not implemented.')

    return new_bboxes


def decompose_rip_images(input_root, output_root):
    img_files, ann_files = load_folder_files(input_root, dicts=['img', 'ann_mask'])

    pbar = tqdm.tqdm(zip(img_files, ann_files))
    for cnt, (img_file, ann_file) in enumerate(pbar):
        frame = img_file.split('/')[-1]
        name, ext = frame.split('.')
        input_img_folder = os.path.dirname(img_file)
        output_img_folder = os.path.join(output_root, input_img_folder)
        output_img_folder = output_img_folder.replace('/img', '/img_patches')
        output_img_folder = os.path.join(output_img_folder, name)
        os.makedirs(output_img_folder, exist_ok=True)

        input_ann_folder = os.path.dirname(ann_file)
        output_ann_folder = os.path.join(output_root, input_ann_folder)
        output_ann_folder = output_ann_folder.replace('/ann_mask', '/ann_mask_patches')
        output_ann_folder = os.path.join(output_ann_folder, name)
        os.makedirs(output_ann_folder, exist_ok=True)

        anno = json.load(open(os.path.join(input_root, ann_file), 'r'))
        img = cv2.imread(os.path.join(input_root, img_file))
        patches = decompose_image(img, anno, patch_size=[800, 800], stride=[300, 700])
        for patch_id, value in patches.items():
            output_img_path = os.path.join(output_img_folder, f'{name}_patch_{patch_id:02d}.png')
            output_ann_path = os.path.join(output_ann_folder, f'{name}_patch_{patch_id:02d}.png.json')
            cv2.imwrite(output_img_path, value.image)
            json.dump(value.annos, open(output_ann_path, 'w'), indent=2)
            pass
        pass

    return


def test_pad_image():
    image = cv2.imread('./tests/img_cv.png')
    output = pad_image(image, [(1920 - 1080) // 2, (1920 - 1080) // 2, 0, 0])
    cv2.imwrite('./tests/img_cv_pad_1.png', output)
    output = pad_image(image, [(2000 - 1080) // 2, (2000 - 1080) // 2, 40, 40])
    cv2.imwrite('./tests/img_cv_pad_2.png', output)
    output = pad_image(image, [(2000 - 1080), 0, 80, 0])
    cv2.imwrite('./tests/img_cv_pad_3.png', output)


def test_decompose_image():
    shutil.rmtree('./tests/image_decompose')
    os.makedirs('./tests/image_decompose', exist_ok=True)
    image = cv2.imread('./tests/img_cv.png')
    bboxes = [[0, 0, 100, 100], [0, 100, 1000, 1000], [100, 100, 200, 200]]
    patch_size = [800, 800]
    stride = [300, 700]
    patches = decompose_image(image, bboxes, patch_size, stride)
    for key, value in patches.items():
        cv2.imwrite(f'./tests/image_decompose/patch_{key}.png', value.image)
        json.dump({'objects': value.bboxes}, open(f'./tests/image_decompose/patch_{key}.png.json', 'w'))
    pass


def test_decompose_rip_images():
    input_root = '/data2/data2/zewei/data/RipData/RipTrainingAllData'
    output_root = '/data2/data2/zewei/data/RipData/RipTrainingAllData'
    decompose_rip_images(input_root, output_root)


def test_assemble_patches():
    bboxes = [0, 0, 100, 100]
    assemble_patches(bboxes, [800, 800], [300, 700], image_size=(1080, 1920), type='xyxy')


if __name__ == '__main__':
    # test_pad_image() # checked
    # test_decompose_image() # checked
    test_decompose_rip_images()
    # convert_folders()
