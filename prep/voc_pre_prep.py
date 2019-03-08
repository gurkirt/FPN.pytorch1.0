import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

## Specify your base directory
base_dir = '/mnt/mars-fast/datasets/voc/'
base_dir = '/mnt/mercury-fast/datasets/voc/'
# base_dir = '/home/gurkirt/voc/'

def convert(size, box):
    '''Normlise box coordinate between 0 and 1'''
    dw = 1./size[0]
    dh = 1./size[1]
    # x = (box[0] + box[1])/2.0
    # y = (box[2] + box[3])/2.0
    # w = box[1] - box[0]
    # h = box[3] - box[2]
    xmin = box[0]*dw
    xmax = box[1]*dw
    ymin = box[2]*dh
    ymax = box[3]*dh
    return (xmin,ymin,xmax,ymax)

def convert_annotation(year, image_id):
    '''Convert annotations to text format xmin ymin xmax ymax label; xmin ymin ......'''
    in_file = open(base_dir+'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open(base_dir+'VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(" ".join([str(a) for a in bb]) + " " + str(float(cls_id)) + '\n')

for year, image_set in sets:
    if not os.path.exists(base_dir + 'VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs(base_dir + 'VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open(base_dir+'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open(base_dir+'%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('VOCdevkit/VOC%s/JPEGImages/%s\n'%(year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val')]

cat_file = 'cat '
for year, image_set in sets:
    cat_file += base_dir + '%s_%s.txt ' % (year, image_set)
print('train file is going to be ', cat_file)
os.system(cat_file+ ' > '+base_dir+'train.txt')
os.system('mv '+base_dir+'2007_test.txt '+base_dir+'test.txt')