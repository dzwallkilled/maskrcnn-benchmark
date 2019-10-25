__author__ = 'zwding'
__version__ = '1.0'

import json
import time
import os
import sys
import argparse
import numpy as np
import cv2

from deep_image_matting.core.matter import Matter, binary_mask_to_polygon, binary_mask_to_rle
from generate_masks import butterfly_mask, gaussian_mask


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--outdir', help="output dir for json files", default='', type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default='', type=str)
    return parser.parse_args()


class RIP:
    categories = {'Flash Rip': 0,
                  'Rip Neck': 1,
                  'Rip Head': 2,
                  'Could be a Flash Rip?': 3,
                  'Could be a rip?': 4,
                  'Rip': 5}

    def __init__(self, path=None, matter=None):
        self.matter = matter
        if path is not None:
            print('loading annotations into memory...')
            tic = time.time()
            self.all_set, self.train_set, self.test_set = self._load_folder(path)
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def _load_folder(self, path):
        ## TODO: seperate the function into two,
        # one to get the whole dataset,
        # the other to divide the dataset into train and test according to whether having annotations
        lists = os.listdir(path)
        lists.sort()
        training_imgs = []
        total_imgs = []
        test_imgs = []
        total_anns = []
        img_id = 0
        ann_id = 0
        for _folder in lists[:-2]:
            _imgs = os.listdir(os.path.join(path, _folder, 'img'))
            _imgs.sort()
            _train_imgs = []
            _test_imgs = []
            _all_imgs = []
            _annotations = []
            for _m in _imgs:
                img_file = os.path.join(_folder, 'img', _m)
                # image = cv2.imread(os.path.join(path, img_file))
                image = None
                ann = json.load(open(os.path.join(path, _folder, 'ann', f'{_m}.json'), 'r'))
                image_obj = {
                    "id": img_id,
                    "width": ann['size']['width'],
                    "height": ann['size']['height'],
                    "file_name": img_file,
                    "license": [],
                }
                _all_imgs.append(image_obj)
                if ann['tags'] or ann['objects']:  # if the image is annotated
                    _train_imgs.append(image_obj)
                    _annotations.extend(self._add_annotation(ann_id, img_id, ann['objects'], image))
                    ann_id += len(ann['objects'])
                else:
                    _test_imgs.append(image_obj)
                img_id += 1

            print(f'loading folder {_folder}: {len(_all_imgs)} images, '
                  f'{len(_train_imgs)} images annotated, '
                  f'{len(_test_imgs)} images not annotated,  '
                  f'{len(_annotations)} annotations.')
            training_imgs += _train_imgs
            total_anns += _annotations
            total_imgs += _all_imgs
            test_imgs += _test_imgs
            pass

        print(f'Total {len(lists[:-2])} folders with {len(total_imgs)} images\n'
              f'{len(training_imgs)} images annotated with {len(total_anns)} annotations\n'
              f'{len(test_imgs)} images NOT annotated')
        train_dataset = self._make_dataset(training_imgs, 'RIP data with annotations', total_anns)
        test_dataset = self._make_dataset(test_imgs, 'RIP data without annotations')
        all_dataset = self._make_dataset(total_imgs, 'RIP all data', total_anns)
        return all_dataset, train_dataset, test_dataset

    def _add_annotation(self, ann_id: int, image_id: int, ann: dict, image=None):
        annotations = []
        for _id, _obj in enumerate(ann):
            x1, y1 = map(float, _obj['points']['exterior'][0])
            x2, y2 = map(float, _obj['points']['exterior'][1])
            left, right = min(x1, x2), max(x1, x2)
            bottom, top = max(y1, y2), min(y1, y2)
            width = right - left
            height = bottom - top
            bbox = [left, top, width, height]
            category_id = self.categories[_obj['classTitle']]
            iscrowd = 1 if len(ann) > 1 and category_id in [1, 2] else 0
            segmentation = self._bbox_to_segmentation(image, bbox) if image is not None else None
            _ann = {
                "id": ann_id + _id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": width * height,
                "bbox": bbox,
                "iscrowd": iscrowd,
            }
            annotations.append(_ann)
        return annotations

    def _bbox_to_segmentation(self, image, bbox, seg_type='rle', mask_type='gaussian', matting=True):
        """

        :param image: ndarry, (img_h, img_w, 3)
        :param bbox: [int], x, y, w ,h
        :param seg_type: 'rle', 'poly'
        :param mask_type: 'gaussian', 'bbox', 'butterfly'
        :param matting: bool, whether to mat the mask or not
        :return:
        """
        img_shape = image.shape[:2]
        if mask_type == 'bbox':
            mask = np.zeros(img_shape, dtype=np.uint8)
        elif mask_type == 'gaussian':
            mask = gaussian_mask(bbox, )
        elif mask_type == 'butterfly':
            mask = butterfly_mask(bbox)
        else:
            raise ValueError('No such mask type')

        if matting:
            mask = self.matter.matting(image, mask)

        mask = mask > 25

        if seg_type == 'polygon':
            segmentation = binary_mask_to_polygon(mask, tolerance=2)
        elif seg_type == 'rle':
            segmentation = binary_mask_to_rle(mask)
        else:
            raise ValueError(f'{seg_type} not defined. (polygon/rle)')

        return segmentation

    def _make_dataset(self, imgs, description, annotations=None):
        dataset = dict()
        dataset['images'] = imgs
        if annotations:
            dataset['annotations'] = annotations
        dataset['info'] = dict(year=2019,
                               version=__version__,
                               description=description,
                               contributor=__author__,
                               date_created=2019,
                               url=None)
        dataset['liscences'] = []
        dataset['categories'] = self._add_categories()
        return dataset

    def _add_categories(self):
        categories = []
        for key, value in self.categories.items():
            category = {'id': value,
                        'name': key,
                        'supercategory': 'Beach'}
            categories.append(category)
        return categories

    def save_dataset(self, path, dataset=None):
        """

        :param path:
        :param dataset:
        :return:
        """
        path = os.path.abspath(os.path.expanduser(path)).split('.')[0]
        os.makedirs(path, exist_ok=True)
        print(f'saving to path {os.path.dirname(path)}')
        tic = time.time()
        if dataset is not None:
            json.dump(open(path), dataset)
        else:
            json.dump(self.all_set, open(f'{path}/rip_data_all.json', 'w'), )
            json.dump(self.train_set, open(f'{path}/rip_data_train.json', 'w'))
            json.dump(self.test_set, open(f'{path}/rip_data_test.json', 'w'))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def save_dataset_kfold(self, path, k=5, seed=123):
        """

        :param path: folder path to save the datasets
        :param k: number of folds for k-fold cross validation, default 5, 80%/20%
        :return:
        """
        folder = os.path.join(path, f'cv_{k}_folder')
        os.makedirs(folder, exist_ok=True)

        np.random.seed(seed)

        total_set = self.train_set
        num_sample = len(total_set['images'])
        perm_index = np.random.permutation(num_sample)
        fold = num_sample // k
        for _k in range(k):
            val_index = perm_index[fold * _k: fold * (_k + 1)]
            train_index = [i for i in perm_index if not i in val_index]
            self._save_cv_dataset(train_index, os.path.join(folder, f'train_{_k + 1}.json'),
                                  f"K-fold cross validation TRAIN {_k + 1}/{k}")
            self._save_cv_dataset(val_index, os.path.join(folder, f'val_{_k + 1}.json'),
                                  f"K-fold cross validation VAL {_k + 1}/{k}")

    def _save_cv_dataset(self, indexes, path, description):
        cv_images = [self.train_set['images'][i] for i in indexes]
        cv_img_idx = [img['id'] for img in cv_images]
        cv_annos = [a for a in self.train_set['annotations'] if a['image_id'] in cv_img_idx]
        cv_set = self._make_dataset(cv_images, description, cv_annos)
        json.dump(cv_set, open(path, 'w'))
        print(path)


def save_dataset():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parse_args()
    # path = '/Volumes/DZW-13656676703/RipData/RipTrainingAllData'
    args.datadir = '/data2/data2/zewei/data/RipData/RipTrainingAllData'
    args.outdir = '/data2/data2/zewei/data/RipData/'
    matter = Matter(resume='./deep_image_matting/model/stage1_sad_54.4.pth')
    dataset = RIP(path=args.datadir, matter=matter)
    # dataset.save_dataset(args.outdir)
    # dataset.save_dataset_kfold(path=args.outdir, k=5)


def view_data():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parse_args()
    args.datadir = '/data2/data2/zewei/data/RipData/RipTrainingAllData'
    args.outdir = '/data2/data2/zewei/data/RipData/'
    from pycocotools.coco import COCO
    coco = COCO(annotation_file=os.path.join(args.outdir, 'rip_data_train.json'))
    anns = coco.loadAnns(0)
    coco.showAnns(anns)
    pass


if __name__ == '__main__':
    save_dataset()
    # view_data()
