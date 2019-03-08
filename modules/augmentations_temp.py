import torch, pdb
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import types
from numpy import random

DEBUG_AUG = False

def intersect(box_a, box_b):

    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):

    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """

    #if DEBUG_AUG:
        # print('jaccard_numpy()')

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, boxes=None, labels=None, seq_len=1):

        for t in self.transforms:
            imgs, boxes, labels = t(imgs, boxes, labels, seq_len)

        return imgs, boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('ToAbsoluteCoords()')

        width, height = images.size
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return images, boxes, labels


class ToPercentCoords(object):
    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('ToPercentCoords()')

        width, height = images.size
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return images, boxes, labels

class RandomSampleCrop(object):

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None, seq_len=None):

        width, height = image.size
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                #current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap coniou.max() <= max_ioustraint satisfied? if not try again
                if overlap.min() < min_iou or overlap.max() > max_iou:
                    continue

                # cut the crop from the image
                #current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                cropped_imgs = []
                for i in range(seq_len):
                    # cut the crop from the images
                    current_image = image[i, :, :, :]
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                    cropped_imgs += [current_image]
                cropped_imgs = np.array(cropped_imgs)

                return cropped_imgs, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, images, boxes, labels, seq_len=None):

        if random.randint(2):
           return images, boxes, labels

        num_imgs, height, width, depth = images.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        exp_imgs = np.zeros((num_imgs, int(height*ratio), int(width*ratio), depth), dtype=images.dtype)
        for i in range(num_imgs):
            exp_imgs[i, :, :, :] = self.mean
            exp_imgs[i, int(top):int(top + height), int(left):int(left + width)] = images[i, :, :, :]
            # image = exp_imgs

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return exp_imgs, boxes, labels


class RandomMirror(object):
    def __call__(self, images, boxes, classes, seq_len=None):

        if DEBUG_AUG:
            print('RandomMirror()')

        if random.random() < 0.5:
            images = images.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes is not None:
                xmin = 1 - boxes[:, 2]
                xmax = 1 - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
        return images, boxes, classes


class random_distort(object):
    def __init__(self):
        self.brightness_delta=32/255.,
        self.contrast_delta=0.5,
        self.saturation_delta=0.5,
        self.hue_delta=0.1

    def __call__(self, images, boxes, classes, seq_len=None):
        images = self._brightness(images, self.brightness_delta)
        if random.random() < 0.5:
            images = self._contrast(images, self.contrast_delta)
            images = self._saturation(images, self.saturation_delta)
            images = self._hue(images, self.hue_delta)
        else:
            images = self._saturation(images, self.saturation_delta)
            images = self._hue(images, self.hue_delta)
            images = self._contrast(images, self.contrast_delta)
        return images,  boxes, classes

    def _brightness(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(brightness=delta)(img)
        return img

    def _contrast(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(contrast=delta)(img)
        return img

    def _saturation(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(saturation=delta)(img)
        return img

    def _hue(img, delta):
        if random.random() < 0.5:
            img = transforms.ColorJitter(hue=delta)(img)
        return img


#-----------------------------------------------------------------------------
# ----------------------------------- NOTE -----------------------------------
'''
ConvertFromInts:        convert the img type from int to float
ToAbsoluteCoords:       convert the normalised coordinates to absolute/original coordinates  
Expand:                 expand the image and gt boxes at random with prob 0.5, it is different from Resize(), it expand the image (expand/shrink) within WxH diemnsion
RandomSampleCrop:       crop an patch from img at random within 4/5 given option/choices and adjust the boxes coordinates accordingly -- this function may remove some boxes, careful with multiple instance
RandomMirror:           horizontal flip of img and boxes at random with prob 0.5
ToPercentCoords:        normalised the coordinates back again for training
Resize:                 resize the img and boxes as per 300x300 dim
SubtractMeans:          subtract image mean

'''
#-----------------------------------------------------------------------------

class SSDAugmentation(object):
    def __init__(self, size=600, means=(104, 117, 123), std= ()):

        self.mean = mean
        self.size = size

        self.augment = Compose([
            ToAbsoluteCoords(),
            random_distort(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            #Resize(self.size),
            #SubtractMeans(self.mean),
        ])

    def __call__(self, imgs, boxes, labels, seq_len):
        return self.augment(imgs, boxes, labels, seq_len) # calling the Compose()s
