__author__ = 'zwding'
__version__ = '1.0'

import json
import time
import os
import argparse
import numpy as np
import cv2
import tqdm
from collections import defaultdict
from pycocotools import mask as maskUtils
from deep_image_matting.core.matter import Matter, binary_mask_to_polygon, binary_mask_to_rle
from generate_masks import butterfly_mask, gaussian_mask

from pycocotools.coco import COCO


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


def load_folder_files(path, dicts=[]):
    if not path.endswith('/'):
        path = path + '/'
    folders = next(os.walk(path))[1]
    folders.sort()
    total_files = []
    for subfolder in dicts:
        files = []
        for _folder in folders:
            for r, f, _files in os.walk(os.path.join(path, _folder, subfolder)):
                _files = list(map(lambda _m: os.path.join(r.replace(path, ''), _m), _files))
                files.extend(_files)
        files.sort()
        print(f'Total {len(folders)} folders with {len(files)} images')
        total_files.append(files)
    return total_files


def bbox_to_segmentation(image, bbox, matter=None, seg_type='rle', mask_type='gaussian'):
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
        mask = gaussian_mask(bbox, imgh=img_shape[0], imgw=img_shape[1])
    elif mask_type == 'butterfly':
        mask = butterfly_mask(bbox)
    else:
        raise ValueError('No such mask type')

    if matter is not None:
        mask = matter.matting(image, mask)

    if seg_type == 'polygon':
        segmentation = binary_mask_to_polygon(mask > 10, tolerance=2)
    elif seg_type == 'rle':
        mask[mask > 10] = 255
        mask[mask <= 10] = 0
        segmentation = maskUtils.encode(np.asfortranarray(mask))
    else:
        raise ValueError(f'{seg_type} not defined. (polygon/rle)')

    return segmentation


def add_categories(categories):
    cat_obj = []
    for key, value in categories.items():
        category = {'id': value,
                    'name': key,
                    'supercategory': 'Beach'}
        cat_obj.append(category)
    return cat_obj


class RIP:
    categories = {'Flash Rip': 0,
                  'Rip Neck': 1,
                  'Rip Head': 2,
                  'Could be a Flash Rip?': 3,
                  'Could be a rip?': 4,
                  'Rip': 5}

    def __init__(self, path=None, matter=None, level='level1', anno_type=('bbox',), patches=False):
        self.path = path
        self.level = level
        self.anno_type = anno_type
        self.matter = matter
        self.img_files = []
        self.anno_files = []
        self.patches = patches
        if 'seg' in anno_type:
            dicts = ['img', 'ann_mask']
        else:
            dicts = ['img', 'ann']
        if patches:
            dicts = [x+'_patches' for x in dicts]
        dicts = ['img_patches', 'ann_mask_patches_v1']
        if path is not None:
            print('loading annotations into memory...')
            tic = time.time()
            self.img_files, self.anno_files = load_folder_files(path, dicts=dicts)
            self.all_set, self.train_set, self.test_set = self._to_coco()
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def _to_coco(self):
        """
        read all img_files and anno_files into memory,
        and prepare it as COCO format
        :return:
        """
        total_imgs = []
        total_anns = []
        anno_id = 0
        tbar = tqdm.trange(len(self.img_files))
        tbar.set_description_str(f'{self.level}')
        for img_id, (img_file, anno_file) in enumerate(zip(self.img_files, self.anno_files)):
            anno = json.load(open(os.path.join(self.path, anno_file), 'r'))

            total_imgs.append({
                "id": img_id,
                "width": anno['size']['width'],
                "height": anno['size']['height'],
                "file_name": img_file,
                "license": [],
            })

            if anno['objects']:  # if the image is annotated
                annos = self._get_anno_obj(anno_id, img_id, anno['objects'])
                total_anns.extend(annos)
                anno_id += len(annos)
            tbar.update()
            pass
        tbar.close()

        train_img_indexes = defaultdict(int)
        for ann in total_anns:
            train_img_indexes[ann['image_id']] = 1
        train_imgs = [img for img in total_imgs if train_img_indexes[img['id']] == 1]
        test_imgs = [img for img in total_imgs if train_img_indexes[img['id']] == 0]
        print(f'Total images {len(total_imgs)}, '
              f'{len(train_imgs)} annotated with {len(total_anns)} annotations, '
              f'{len(test_imgs)} unannotated')

        all_dataset = self._make_dataset(total_imgs, 'RIP all data', total_anns)
        train_set = self._make_dataset(train_imgs, 'RIP train data', total_anns)
        test_set = self._make_dataset(test_imgs, 'RIP test data')
        return all_dataset, train_set, test_set

    def _get_anno_obj(self, ann_id: int, image_id: int, ann: dict):
        annotations = []
        _id = 0
        for _obj in ann:
            category_id = self.categories[_obj['classTitle']]
            # iscrowd = 1 if len(ann) > 1 and category_id in [1, 2] else 0
            iscrowd = 1  # the segmentation uses RLE format
            bbox = _obj['bbox']
            segmentation = _obj['mask']
            _ann = {
                "id": ann_id + _id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": bbox[2] * bbox[3],
                "bbox": bbox,
                "iscrowd": iscrowd,
            }
            annotations.append(_ann)
            _id += 1
        return annotations

    def _make_dataset(self, imgs, description, annotations=None):
        dataset = {'images': imgs,
                   'info': {'year': 2019,
                            'version': __version__,
                            'description': description,
                            'contributor': __author__,
                            'date_created': 2019,
                            'url': None},
                   'liscences': [],
                   'categories': add_categories(self.categories)}
        if annotations is not None:
            dataset['annotations'] = annotations

        return dataset


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
        os.makedirs(path, exist_ok=True)

        np.random.seed(seed)

        total_set = self.train_set
        num_sample = len(total_set['images'])
        perm_index = np.random.permutation(num_sample)
        fold = num_sample // k
        for _k in range(k):
            val_index = perm_index[fold * _k: fold * (_k + 1)]
            train_index = [i for i in perm_index if not i in val_index]
            self._save_cv_dataset(train_index, os.path.join(path, f'train_{_k + 1}.json'),
                                  f"K-fold cross validation TRAIN {_k + 1}/{k}")
            self._save_cv_dataset(val_index, os.path.join(path, f'val_{_k + 1}.json'),
                                  f"K-fold cross validation VAL {_k + 1}/{k}")

    def _save_cv_dataset(self, indexes, path, description):
        cv_images = [self.train_set['images'][i] for i in indexes]
        cv_img_idx = [img['id'] for img in cv_images]
        cv_annos = [a for a in self.train_set['annotations'] if a['image_id'] in cv_img_idx]
        cv_set = self._make_dataset(cv_images, description, cv_annos)
        json.dump(cv_set, open(path, 'w'))
        print(path)


def save_dataset():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = parse_args()
    # path = '/Volumes/DZW-13656676703/RipData/RipTrainingAllData'
    # args.datadir = '/data2/data2/zewei/data/RipData/RipTrainingAllData'
    # args.outdir = f'/data2/data2/zewei/data/RipData/COCOJSONs/full'
    args.datadir = '/data2/data2/zewei/data/RipData/RipTrainingAllData'
    args.outdir = f'/data2/data2/zewei/data/RipData/COCOJSONPatches_v1/full/'
    matter = Matter(resume='deep_image_matting/model/stage1_sad_54.4.pth', max_size=800)
    dataset = RIP(path=args.datadir, matter=matter, level='full', anno_type=('bbox', 'seg'), patches=True)
    dataset.save_dataset(args.outdir)
    dataset.save_dataset_kfold(path=os.path.join(args.outdir, 'cv_5_fold'), k=5)


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


def convert_full_to_hier(anno_file, output_file, level='level1'):
    cats = {'Flash Rip': 0,
            'Rip Neck': 1,
            'Rip Head': 2,
            'Could be a Flash Rip?': 3,
            'Could be a rip?': 4,
            'Rip': 5}

    categories = {'level1': {'Rip': 0},
                  'level2': {'Rip': 0, 'Flash Rip': 5, },
                  'level3': {'Rip': 5,
                             'Flash Rip': 0,
                             'Rip Neck': 1,
                             'Rip Head': 2},
                  'full': {'Rip': 5,
                           'Flash Rip': 0,
                           'Rip Neck': 1,
                           'Rip Head': 2,
                           'Could be a rip?': 4,
                           'Could be a Flash Rip?': 3}}

    catsMapping = {'level1': {0: 0, 1: None, 2: None, 3: None, 4: None, 5: 0},
                   'level2': {0: 0, 1: None, 2: None, 3: None, 4: None, 5: 5},
                   'level3': {0: 0, 1: 1, 2: 2, 3: None, 4: None, 5: 5}}

    annos = json.load(open(anno_file, 'r'))
    anno_id = 0
    new_annotations = []
    none_count = 0
    for anno in annos['annotations']:
        cat_id = anno['category_id']
        new_cat_id = catsMapping[level][cat_id]
        if new_cat_id is not None:
            anno['category_id'] = new_cat_id
            anno['id'] = anno_id
            anno_id += 1
            new_annotations.append(anno)
        else:
            none_count += 1
    print(f"total annos {len(annos['annotations'])}, removed {none_count}, remained {len(new_annotations)}")
    annos['annotations'] = new_annotations

    annos['categories'] = add_categories(categories[level])
    json.dump(annos, open(output_file, 'w'))
    print(f'saved to {output_file}')
    pass


def convert_data():
    args = parse_args()
    data_dir = '/data2/data2/zewei/data/RipData/'

    for datadir in ['COCOJSONPatches_v1', 'COCOJSONs']:
        args.datadir = data_dir + datadir
        for level in ['level1', 'level2', 'level3']:
            for folder in ['cv_5_fold', 'cv_10_fold']:
                anno_dir = os.path.join(args.datadir, 'full', folder)
                out_dir = os.path.join(args.datadir, level, folder)
                os.makedirs(out_dir, exist_ok=True)
                for _, _, files in os.walk(anno_dir):
                    files = sorted(files)
                    for _file in files:
                        anno_file = os.path.join(anno_dir, _file)
                        out_file = os.path.join(out_dir, _file)
                        print(f'processing {anno_file}')
                        convert_full_to_hier(anno_file, out_file, level)
    pass


if __name__ == '__main__':
    # convert_image_file()
    save_dataset()
    convert_data()
    # view_data()
