import json
import os
import numpy as np
import cv2
from cv2.ximgproc import *


def butterfly_mask(bbox, imgh=1080, imgw=1920):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cx = x + w//2
    cy = y + h//2
    dx0, dy0 = w//16, h // 16
    p = 1
    xs = np.arange(imgw, dtype=np.float32)
    xs = 1 / np.sqrt(1 + (np.clip(np.abs(xs - cx) - w // 8, 0, w) / dx0) ** (2*p))
    xs[:x] = xs[x+w:] = 0

    ys = np.arange(imgh, dtype=np.float32)
    ys = 1 / np.sqrt(1 + (np.clip(np.abs(ys - cy) - h // 8, 0, h) / dy0) ** (2 * p))
    ys[:y] = ys[y+h:] = 0
    xx, yy = np.meshgrid(xs, ys)

    return xx * yy


def gaussian_mask(bbox, imgh=1080, imgw=1920):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cx = x + w // 2
    cy = y + h // 2
    sigma_x = w // 3
    sigma_y = h // 3
    xx, yy = np.meshgrid(np.arange(imgw, dtype=np.float32), np.arange(imgh, dtype=np.float32))
    # mask = np.exp(-0.5*(((xx - cx)*h/w) ** 2 + ((yy - cy)*1) ** 2) / sigma **2)
    mask = np.exp(-0.5 * ((xx - cx)**2/sigma_x ** 2 + (yy-cy)**2/sigma_y**2))
    mask[:y, :] = mask[y+h:, :] = mask[:, :x] = mask[:, x+w:] = 0
    return mask


class MaskSaver:
    def __init__(self, root, anno_file):
        self.root = root
        self.annos = json.load(open(anno_file, 'r'))
        self._len = len(self.annos['images'])
        self._index = 0

    def __getitem__(self, index):
        img = self.annos['images'][index]
        img_id = img['id']
        img_shape = (img['height'], img['width'])
        img_cv = cv2.imread(os.path.join(self.root, img['file_name']))

        bboxes = [ann['bbox'] for ann in self.annos['annotations'] if
                  ann['image_id'] == img_id and ann['category_id'] == 0]
        mask = np.zeros(img_shape, dtype=np.uint8)
        for bbox in bboxes:
            # mask = butter_mask(bbox=bbox)
            mask = gaussian_mask(bbox)
            # bbox = [int(i) for i in bbox]
            # x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 50
            # mask[y + h // 10:y + 9 * h // 10, x + w // 10:x + w * 9 // 10] = 200

        return img_cv, (mask * 255).astype(np.uint8)

    def __len__(self):
        return self._len

    def get_info(self, index):
        return self.annos['images'][index]['id'], \
               self.annos['annotations'][index]['image_id'],\
               self.annos['annotations'][index]['id'], \
               self.annos['annotations'][index]['bbox']

    def step(self, img_path='img_cv.png', mask_path='mask.png', index=None):
        if index is None:
            index = self._index
            self._index += 1
        if index >= self._len:
            index = index % self._len

        img_cv, mask = self.__getitem__(index)
        cv2.imwrite(img_path, img_cv)
        cv2.imwrite(mask_path, mask)

    def step_gif(self, img_path='img_cv.png', mask_path='mask.png', index=None):
        if index is None:
            index = self._index
            self._index = (self._index + 1) % self._len
        if index >= self._len:
            index = index % self._len

        img_cv, mask = self.__getitem__(index)
        cv2.imwrite(img_path, img_cv)
        cv2.imwrite(mask_path, mask)
        guide = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        dst1 = cv2.ximgproc.guidedFilter(
            guide=guide, src=mask, radius=16, eps=50, dDepth=-1)
        dst2 = cv2.ximgproc.guidedFilter(
            guide=guide, src=mask, radius=16, eps=200, dDepth=-1)
        dst3 = cv2.ximgproc.guidedFilter(
            guide=guide, src=mask, radius=16, eps=1000, dDepth=-1)
        cv2.imwrite('dst1.png', dst1)
        cv2.imwrite('dst2.png', dst2)
        cv2.imwrite('dst3.png', dst3)


if __name__ == '__main__':
    root = os.path.expanduser('~/data/RipData/RipTrainingAllData')
    anno_file = os.path.expanduser('~/data/RipData/rip_data_train.json')
    generator = MaskSaver(root, anno_file)
    generator.step()