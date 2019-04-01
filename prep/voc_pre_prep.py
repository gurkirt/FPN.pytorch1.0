
"""
    Author: Gurkirt Singh
    Purpose: resave the annotations in desired (json) format

    Licensed under The MIT License [see LICENSE for details]
    
"""

import xml.etree.ElementTree as ET
import pickle
import os
import cv2
import pdb
from os import listdir, getcwd
from os.path import join
import argparse

parser = argparse.ArgumentParser(description='prepare VOC dataset')
# anchor_type to be used in the experiment
parser.add_argument('--base_dir', default='/home/gurkirt/datasets/voc/', help='Location to root directory for the dataset') 
# /mnt/mars-fast/datasets/

sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

visuals = False

def convert(size, box):
    '''Normlise box coordinate between 0 and 1'''
    dw = 1./size[0]
    dh = 1./size[1]
    xmin = box[0]*dw
    xmax = box[2]*dw
    ymin = box[1]*dh
    ymax = box[3]*dh
    return (xmin,ymin,xmax,ymax)

def convert_annotation(base_dir, year, image_id):
    '''Convert annotations to text format xmin ymin xmax ymax label; xmin ymin ......'''
    in_file = open(base_dir+'VOC%s/Annotations/%s.xml'%(year, image_id))

    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    if visuals:
        image = cv2.imread(base_dir+'VOC%s/JPEGImages/%s.jpg'%(year, image_id))

    image_name = 'VOC%s/JPEGImages/%s'%(year, image_id)
    # assert h == image.shape[0] and w == image.shape[1]
    annos = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls_ = obj.find('name').text
        if cls_ not in classes or int(difficult) == 1:
            continue
        anno = dict()
        anno['cls'] = cls_
        cls_id = classes.index(cls_)
        anno['label'] = cls_id
        xmlbox = obj.find('bndbox')
        
        bb = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
        
        if visuals:
            print(bb)
            image = cv2.rectangle(image,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(0,255,0), thickness=10)
            cv2.imshow('image99', image)
            k = cv2.waitKey(0)
        # pdb.set_trace()
        # print(image.shape, h, w, bb)
        bb = convert((w,h), bb)
        # print(bb)
        condition = True
        for b in bb:
            if b<0 and b>1:
                condition = False 
        
        if not (condition and bb[0]<bb[2] and bb[1]<bb[3]):
            print('we have problem in ', bb, in_file)
        anno['bbox'] = bb
        annos.append(anno)

    return annos, w, h, image_name

if __name__ == '__main__':
    args = parser.parse_args()

    annots = dict()
    for year, image_set in sets:
        set_str = image_set+year
        
        image_ids = open(args.base_dir+'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        print(set_str, len(image_ids))
        for image_id in image_ids:
            img_annos, w, h, image_name = convert_annotation(args.base_dir, year, image_id)
            image_tags = dict()
            image_tags['wh'] = [w, h]
            image_tags['annos'] = img_annos
            image_tags['set'] = set_str
            if image_name not in annots.keys():
                annots[image_name] = image_tags
            else:
                print('imagename exists do somthing!!!!!!!!')
    db = dict()
    db['classes'] = classes
    db['annotations'] = annots

    ic = 0
    ac = 0
    for img_id in db['annotations'].keys():
        num_a = len(db['annotations'][img_id]['annos'])
        assert num_a>0, num_a
        ic += 1
        ac += num_a
    print('Avergage number of annotation per image are ', float(ac)/ic)

    import json
    with open(args.base_dir + 'annots.json', 'w') as f:
        json.dump(db,f)