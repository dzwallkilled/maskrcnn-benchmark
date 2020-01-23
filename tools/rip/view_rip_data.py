import json
import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from cv2.ximgproc import *
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


class COCOExt(COCO):
    def __init__(self, data_root='', annotation_file=None):
        super(COCOExt, self).__init__(annotation_file=annotation_file)
        self.data_root = data_root
        # used to make colors for each class
        # self.colors = [2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1]
        self.colors = [[128, 0, 128],
                       [0, 128, 0],
                        [128, 128, 0],
                       [128, 128, 128],
                       [0, 128, 256],
                       [128, 256, 0]]

    def showBBox(self, index=0, thickness=1):
        img = self.loadImgs(index)[0]
        anns = self.loadAnns(self.getAnnIds(imgIds=index))
        img_cv = cv2.imread(os.path.join(self.data_root, img['file_name']))
        cv2.imwrite('./tests/rip_cv.png', img_cv)
        for cnt, ann in enumerate(anns):
            bbox = list(map(lambda m: int(m), ann['bbox']))
            top_left = tuple(bbox[:2])
            # color = tuple(map(lambda m: int(m*(ann['category_id'] + 1)%255), self.colors))
            color = self.colors[ann['category_id']]
            bottom_right = tuple([bbox[0]+bbox[2], bbox[1]+bbox[3]])
            img_cv = cv2.rectangle(
                img_cv, top_left, bottom_right, tuple(color), thickness
            )
            pass
        cv2.imwrite('./tests/rip_bbox.png', img_cv)
        pass


if __name__ == '__main__':
    data_root = '/data2/data2/zewei/data/RipData/RipTrainingAllData/'
    anno_file = os.path.expanduser('~/data/RipData/COCOJSONs/full/rip_data_train.json')
    dataset = COCOExt(data_root, anno_file)
    dataset.showBBox(0, thickness=3)