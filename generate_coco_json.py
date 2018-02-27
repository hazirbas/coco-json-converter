from __future__ import division
from __future__ import print_function

import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi


class DAVIS2016():
    """
        DAVIS 2016 class to convert annotations to COCO Json format
    """
    def __init__(self, datapath, imageres="480p"):
        self.info = {"year" : 2016,
                     "version" : "1.0",
                     "description" : "A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation (DAVIS)",
                     "contributor" : "F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, A. Sorkine-Hornung ",
                     "url" : "http://davischallenge.org/",
                     "date_created" : "2016"
                    }
        self.licenses = [{"id": 1,
                          "name": "Attribution-NonCommercial",
                          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                         }]
        self.type = "instances"
        self.datapath = datapath
        self.seqs = yaml.load(open(os.path.join(self.datapath, "Annotations", "db_info.yml"),
                                   "r")
                             )["sequences"]

        self.categories = [{"id": seqId+1, "name": seq["name"], "supercategory": seq["name"]}
                              for seqId, seq in enumerate(self.seqs)]
        self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}

        for s in ["train", "trainval", "val"]:
            imlist = np.genfromtxt( os.path.join(self.datapath, "ImageSets", imageres, s + ".txt"), dtype=str)
            images, annotations = self.__get_image_annotation_pairs__(imlist)
            json_data = {"info" : self.info,
                         "images" : images,
                         "licenses" : self.licenses,
                         "type" : self.type,
                         "annotations" : annotations,
                         "categories" : self.categories}

            with open(os.path.join(self.datapath, "Annotations", imageres + "_" +
                                   s+".json"), "w") as jsonfile:
                json.dump(json_data, jsonfile, sort_keys=True, indent=4)

    def __get_image_annotation_pairs__(self, image_set):
        images = []
        annotations = []
        for imId, paths in enumerate(image_set):
            impath, annotpath = paths[0], paths[1]
            print (impath)
            name = impath.split("/")[3]
            img = np.array(Image.open(os.path.join(self.datapath + impath)).convert('RGB'))
            mask = np.array(Image.open(os.path.join(self.datapath + annotpath)).convert('L'))
            if np.all(mask == 0):
                continue

            segmentation, bbox, area = self.__get_annotation__(mask, img)
            images.append({"date_captured" : "2016",
                           "file_name" : impath[1:], # remove "/"
                           "id" : imId+1,
                           "license" : 1,
                           "url" : "",
                           "height" : mask.shape[0],
                           "width" : mask.shape[1]})
            annotations.append({"segmentation" : segmentation,
                                "area" : np.float(area),
                                "iscrowd" : 0,
                                "image_id" : imId+1,
                                "bbox" : bbox,
                                "category_id" : self.cat2id[name],
                                "id": imId+1})
        return images, annotations

    def __get_annotation__(self, mask, image=None):

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        RLE = cocomask.merge(RLEs)
        # RLE = cocomask.encode(np.asfortranarray(mask))
        area = cocomask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(mask)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.drawContours(image, contours, -1, (0,255,0), 1)
            cv2.rectangle(image,(x,y),(x+w,y+h), (255,0,0), 2)
            cv2.imshow("", image)
            cv2.waitKey(1)

        return segmentation, [x, y, w, h], area

if __name__ == "__main__":
    datapath = "/work/datasets/DAVIS-2016/"
    DAVIS2016(datapath)

    # test
    from PIL import Image
    from pycocotools import coco; c = coco.COCO(datapath+'/Annotations/480p_trainval.json')
    Image.fromarray(c.annToMask(c.loadAnns([255])[0])*255).show()
