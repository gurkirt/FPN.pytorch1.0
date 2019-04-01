
"""
    Author: Gurkirt Singh
    Purpose: resave the annotations in desired format

    Licensed under The MIT License [see LICENSE for details]
    
"""

import json, os, pdb, cv2

import argparse

parser = argparse.ArgumentParser(description='prepare VOC dataset')
# anchor_type to be used in the experiment
parser.add_argument('--base_dir', default='/home/gurkirt/datasets/voc/', help='Location to root directory for the dataset') 
# /mnt/mars-fast/datasets/

visuals = False

def get_wh(images):
    img_list = {}
    for img in images:
        image_id = img['id']
        box = [img['width'], img['height']]
        if str(image_id) in img_list.keys():
            raise ('image_id should not repeat')
        else:
            img_list[str(image_id)] = box

    print('Number of images ', len(img_list))
    return img_list

def convert(size, box):
    '''Normlise box coordinate between 0 and 1'''
    dw = 1./size[0]
    dh = 1./size[1]
    xmin = box[0]*dw
    xmax = box[2]*dw
    ymin = box[1]*dh
    ymax = box[3]*dh
    return (xmin,ymin,xmax,ymax)

def get_coco_classes(anno_file):
    with open(anno_file, 'r') as f:
        obj = json.load(f)
    cls_dict = obj['categories']
    idsto = {}
    cls_list = []
    id_list = []
    super_list = []
    scount = 0
    count = 0
    for c in cls_dict:
        cls_list.append(c['name'])
        id_list.append(str(c['id']))
        supername = c['supercategory']

        if supername not in super_list:
            super_list.append(supername)
            scount += 1
        
        sl = super_list.index(supername)
        idsto[str(c['id'])] = [count, sl]
        # sl = slcount
        count += 1

    return cls_list, super_list, id_list, idsto

def get_image_annots(base_dir, filename, subset_str = 'train2017', annots=dict()):

    with open(filename, 'r') as f:
        obj = json.load(f)
    
    # pdb.set_trace()
    whs = get_wh(obj['images'])
    annos = obj['annotations']

    for anno in annos:
        image_id = anno['image_id']
        str_id = subset_str+'/{:012d}'.format(image_id)
        wh  = whs[str(image_id)]
        if str_id not in annots['annotations'].keys():
            annots['annotations'][str_id] = dict()
            annots['annotations'][str_id]['annos'] = []
            annots['annotations'][str_id]['wh'] = wh
            annots['annotations'][str_id]['set'] = subset_str
        bb = anno['bbox']
        bb[2] = bb[2] + bb[0]
        bb[3] = bb[3] + bb[1]
        cid = anno['category_id']
        labels = annots['idstolabels'][str(cid)]

        
        if visuals:
            imagename = base_dir+subset_str+'/{:012d}.jpg'.format(image_id)
            print(imagename)
            image = cv2.imread(imagename)
            assert wh[0] == image.shape[1] and wh[1] == image.shape[0]
            print(bb, image.shape, wh, annots['classes'][labels[0]] )
            image = cv2.rectangle(image,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(0,255,0), thickness=3)
            cv2.imshow('image99', image)
            k = cv2.waitKey(0)

        bb = convert(wh, bb)
        condition = True
        for b in bb:
            if b<0 and b>1:
                condition = False 
        
        if condition and bb[0]<bb[2] and bb[1]<bb[3]:
            an_b = {}
            an_b['cls'] = annots['classes'][labels[0]]
            an_b['label'] = labels[0]
            an_b['suprlbl'] = labels[1]
            an_b['bbox'] = bb
            annots['annotations'][str_id]['annos'].append(an_b)
        else:
            print('we are skipping ', bb, image_id, len(annots['annotations'][str_id]))
        # annos.append(anno)
  
    return annots

if __name__ == '__main__':
    args = parser.parse_args()

    train_filename = args.base_dir + 'annotations/instances_train2017.json'
    val_filename = args.base_dir + 'annotations/instances_val2017.json'
    
    cls_list, super_list, id_list, idstolabels = get_coco_classes(val_filename)

    db = dict()
    db['classes'] = cls_list
    db['superclasses'] = super_list
    db['ids'] = id_list
    db['idstolabels'] = idstolabels
    db['annotations'] = dict()

    db = get_image_annots(args.base_dir, val_filename, 'val2017', db)
    db = get_image_annots(args.base_dir, train_filename, 'train2017', db)
    ic = 0
    ac = 0
    for img_id in db['annotations'].keys():
        num_a = len(db['annotations'][img_id]['annos'])
        assert num_a>0, num_a
        ic += 1
        ac += num_a
        
    print('Avergage number of annotation per image are ', float(ac)/ic)
    with open(args.base_dir + 'annots.json', 'w') as f:
        json.dump(db,f)

     
    


