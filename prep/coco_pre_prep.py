import json, os, pdb
basedir = '/mnt/mars-fast/datasets/coco/'


def get_wh(images):
    img_list = {}
    for img in images:
        image_id = img['id']
        box = [img['width']]
        box.append(img['height'])
        if str(image_id) in img_list.keys():
            raise ('image_id should not repeat')
        else:
            img_list[str(image_id)] = box
    print('Number of images ', len(img_list))
    return img_list


def write_txt_labels(filename, val=False):
    save_dir = basedir + 'labels/'
    annot_list = {}
    with open(filename, 'r') as f:
        obj = json.load(f)
    whs = get_wh(obj['images'])
    annos = obj['annotations']
    for anno in annos:
        image_id = anno['image_id']
        # print(type(anno['bbox']), type(anno['category_id']))
        box = anno['bbox']
        box.append(anno['category_id'])
        # print(box)
        if str(image_id) in annot_list.keys():
            # print('annot_list[str(image_id)] ', annot_list[str(image_id)])
            annot_list[str(image_id)].append(box)
        else:
            annot_list[str(image_id)] = [box]

    print('Number of images with annotations ', len(annot_list))

    # pdb.set_trace()
    ids = annot_list.keys()
    pretxt = 'train2017'
    if val:
        pretxt = 'val2017'
        ids = whs.keys()

    new_ids = []
    for image_id in ids:
        new_ids.append('{:s}/{:012d}'.format(pretxt,int(image_id)))
        lfile = '{:s}{:012d}.txt'.format(save_dir,int(image_id))
        wh  = whs[image_id]
        #if not os.path.isfile(lfile):
        if image_id in annot_list.keys():
            fid = open(lfile,'w')
            # print(image_id, annot_list[image_id])
            for box in annot_list[image_id]:
                bbox = [box[0]/wh[0], box[1]/wh[1], (box[0] + box[2])/wh[0], (box[1] + box[3])/wh[1]]
                check = True
                for b in bbox:
                    if b<0 or b>1.0001:
                        check = False
                assert check ,'box bound check fails '+' '.join([str(b) for b in bbox])

                fid.write('{:f} {:f} {:f} {:f} {:d}\n'.format(bbox[0], bbox[1], bbox[2], bbox[3], box[4]))
    # else:
        #     raise 'there cant be two file with same annot'+lfile
    return new_ids


def write_list(filename, img_list):
    fid = open(filename,'w')
    for img in img_list:
        fid.write(img+'\n')


if __name__ == '__main__':

    train_filename = basedir + 'instances_train2017.json'
    img_list = write_txt_labels(train_filename)
    write_list(basedir + 'train.txt', img_list)
    val_filename = basedir + 'instances_val2017.json'
    img_list = write_txt_labels(val_filename, val=True)
    write_list(basedir + 'test.txt', img_list)

