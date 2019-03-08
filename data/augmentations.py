"""
# orignally from https://github.com/amdegroot/ssd.pytorch
# Modified by: Gurkir Singh to have PIL image like output and normlisation

"""

import torch, copy
import cv2
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

        if DEBUG_AUG:
            print('Compose()')

        for t in self.transforms:
            imgs, boxes, labels = t(imgs, boxes, labels, seq_len)
        return imgs, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('Lambda()')

        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('ConvertFromInts()')

        return images.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, means, stds):
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('SubtractMeans()')

        images = images.astype(np.float32)

        for i in range(seq_len):
            images[i, :, :, :] -= self.means
            images[i, :, :, :] /= self.stds

        return images.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('ToAbsoluteCoords()')

        num_imgs, height, width, channels = images.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return images, boxes, labels


class ToPercentCoords(object):
    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('ToPercentCoords()')

        _, height, width, _ = images.shape

        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return images, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('Resize()')

        _, height, width, _ = images.shape

        res_imgs = []
        for i in range(seq_len):
            res_imgs += [cv2.resize(images[i, :, :, :], (self.size, self.size))]

            # # pdb.set_trace()
            # # print(np.min(images[i, :, :, :]), np.max(images[i, :, :, :]), np.std(images[i, :, :, :]), np.mean(images[i, :, :, :]), type(images[i, :, :, :]))
            # img = np.uint8(images[i, :, :, :])
            # # print(np.min(img), np.max(img), np.std(img), np.mean(img), img.shape)
            # img = Image.fromarray(img)
            # img = img.resize((self.size, self.size), Image.BILINEAR)
            # img = np.asarray(img)
            # res_imgs += [img]

        res_imgs = np.array(res_imgs)

        return res_imgs, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        rand_sat = random.uniform(self.lower, self.upper)
        if random.randint(2):
            num_imgs = images.shape[0]
            for i in range(num_imgs):
                images[i, :, :, 1] *= rand_sat
        return images, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        rand_hue = random.uniform(-self.delta, self.delta)
        if random.randint(2):
            num_imgs = images.shape[0]
            for i in range(num_imgs):
                images[i, :, :, 0] += rand_hue
                images[i, :, :, 0][images[i, :, :, 0] > 360.0] -= 360.0
                images[i, :, :, 0][images[i, :, :, 0] < 0.0] += 360.0

        return images, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('RandomLightingNoise()')

        if random.randint(2):
            swaps = self.perms[random.randint(len(self.perms))]
            num_imgs = images.shape[0]
            #pdb.set_trace()
            # print(images.shape)
            for i in range(num_imgs):
                img = images[i]
                img = img[:, :, swaps]
                # print(img.shape)
                images[i, :, :, :] = img

        return images, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        num_imgs = images.shape[0]  # images = [num_imgs x H x W x C]
        for i in range(num_imgs):
            if self.current == 'BGR' and self.transform == 'HSV':
                images[i, :, :, :] = cv2.cvtColor(images[i, :, :, :], cv2.COLOR_BGR2HSV)
            elif self.current == 'HSV' and self.transform == 'BGR':
                images[i, :, :, :] = cv2.cvtColor(images[i, :, :, :], cv2.COLOR_HSV2BGR)
            else:
                raise NotImplementedError
        return images, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            images *= alpha

        return images, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            images += delta

        return images, boxes, labels



class CV2toImage(object):
    def __call__(self, images, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('CV2toImage()')
        num_imgs = images.shape[0]  # images = [num_imgs x H x W x C]
        for i in range(num_imgs):
            images[i, :, :, :] = cv2.cvtColor(images[i, :, :, :], cv2.COLOR_BGR2RGB) / 255.0

        return images, boxes, labels

class ToTensor(object):  # TODO: not used anywhere
    def __call__(self, cvimage, boxes=None, labels=None, seq_len=None):

        if DEBUG_AUG:
            print('ToTensor()')

        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


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

        _, height, width, _ = image.shape
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
        self.mean = np.asarray(copy.deepcopy(mean))
        self.mean *= 256.0
        tempmean = copy.deepcopy(self.mean)
        self.mean[0] = tempmean[2]
        self.mean[2] = tempmean[0]
        print('Expand intialised with means', self.mean)

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

        _, _, width, _ = images.shape  # [num_imgs x H x W x C]


        if random.randint(2):
            flipped_imgs = []
            for i in range(seq_len):
                flipped_imgs += [images[i, :, ::-1]]

            images = np.array(flipped_imgs)
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]


        return images, boxes, classes


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, images, boxes, labels, seq_len=None):

        if DEBUG_AUG:
            print('PhotometricDistort()')

        ims = images.copy()
        ims, boxes, labels = self.rand_brightness(ims, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        ims, boxes, labels = distort(ims, boxes, labels)
        return self.rand_light_noise(ims, boxes, labels)



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
SubtractMeans:          subtract image mean and stds divs

'''
#-----------------------------------------------------------------------------

class SSDAugmentation(object):
    def __init__(self, size, means, stds):
        print('SSDAugmentation_v1 : __init__() done!')
        self.means = copy.deepcopy(means)
        self.stds = stds
        self.size = size

        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.means),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            CV2toImage(),
            SubtractMeans(self.means, self.stds)
        ])

    def __call__(self, imgs, boxes, labels, seq_len):
        return self.augment(imgs, boxes, labels, seq_len) # calling the Compose()s
