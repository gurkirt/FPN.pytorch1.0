import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from data.augmentations import Augmentation
from data import BaseTransform
import cv2

class LoadImage(object):

    def __init__(self, space='BGR'):
        self.space = space

    def __call__(self, path_img):

        return cv2.imread(path_img)

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

# means = [0.485, 0.456, 0.485]
# stds = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

image_name = '/home/gurkirt/pic.jpg'

input_dim = 300
img_loader = LoadImage
imgs = np.asarray([cv2.imread(image_name)])
print(type(imgs), imgs.shape, '55555')
# imgs = Image.open(image_name)
# if imgs.mode != 'RGB':
#    imgs = imgs.convert('RGB')
# imgs = np.asarray([np.asarray(imgs)])
ssd_transform = BaseTransform(input_dim, means, stds)
targets = np.asarray([[0,0,1,1,1], [0,0,1,1,1]])
print(np.min(imgs), np.max(imgs), np.std(imgs), np.mean(imgs))
imgs, boxes, labels = ssd_transform(imgs, targets[:, :4], targets[:, 4], 1)
# print(imgs[0].size(), '   44444 ')
cvimage = imgs[0] #
# cvimage = torch.FloatTensor(torch.from_numpy(imgs[0]).permute(2, 0, 1))

img = Image.open(image_name)
if img.mode != 'RGB':
   img = img.convert('RGB')

img = img.resize((input_dim, input_dim), Image.BILINEAR)
plimg = transform(img)

print(plimg.size(), plimg.type(), cvimage.size(), cvimage.type())
print(torch.std(plimg), torch.std(cvimage))
print(torch.mean(plimg), torch.mean(cvimage))
print(torch.sum(torch.abs(plimg-cvimage)))
print(cvimage[:,:1,:1])
print(plimg[:,:1,:1])
print(torch.max(plimg[:,:,:]-cvimage[:,:,:]))
