#from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .detectionDatasets import Detection, detection_collate
import cv2, pdb
import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
def base_transform_nimgs(images, size, mean, stds, seq_len):
    res_imgs = []
    # print(images.shape)
    for i in range(seq_len):
        # img = Image.fromarray(images[i,:, :, :])
        # img = img.resize((size, size), Image.BILINEAR)
        img = cv2.resize(images[i, :, :, :], (size, size)).astype(np.float32)
        #img = images[i, :, :, :].astype(np.float32)
        # img  = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        res_imgs += [img]

    res_imgs = np.asarray(res_imgs)

    for i in range(seq_len):
        res_imgs[i, :, :, :] -= mean
        res_imgs[i, :, :, :] /= stds

    res_imgs = res_imgs.astype(np.float32)

    return res_imgs


class BaseTransform:
    def __init__(self, size, means, stds):
        self.size = size
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, seq_len=1):
        return base_transform_nimgs(image, self.size, self.means, self.stds, seq_len=seq_len), boxes, labels

