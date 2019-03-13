"""
UCF24 Dataset Classes
Author: Gurkirt Singh

Updated by Gurkirt Singh for ucf-24 , MSCOCO, VOC datasets

FOV VOC:
Original author: Francisco Massa for VOC dataset
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot for VOC dataset

Updated by: Gurkirt Singh to accpt text annotations for voc

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width

"""

import os , json
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np

CLASSES = dict()

CLASSES['ucf24'] = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
                    'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
                    'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
                    'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']

CLASSES['daly'] = ['ApplyingMakeUpOnLips', 'BrushingTeeth', 'CleaningFloor', 'CleaningWindows', 'Drinking',
                   'FoldingTextile', 'Ironing', 'Phoning', 'PlayingHarmonica', 'TakingPhotosOrVideos']

CLASSES['coco'] = []

CLASSES['voc'] = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']

def readsplitfile(splitfile):
    with open(splitfile, 'r') as f:
        temptrainvideos = f.readlines()
    trainvideos = []
    for vid in temptrainvideos:
        vid = vid.rstrip('\n')
        trainvideos.append(vid)
    return trainvideos


def make_lists_ucf(rootpath, imgtype, split=1, use_bg=False, fulltest=False):
    imagesDir = rootpath + imgtype + '/'
    splitfile = rootpath + 'splitfiles/trainlist{:02d}.txt'.format(split)
    trainvideos = readsplitfile(splitfile)
    trainlist = []
    testlist = []

    with open(rootpath + 'splitfiles/pyannot.pkl','rb') as fff:
        database = pickle.load(fff)

    train_action_counts = np.zeros(len(CLASSES['ucf24']), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES['ucf24']), dtype=np.int32)
    
    ratios = np.asarray([1.1,1.,4.7,1.4,1.,2.6,2.2,3.0,3.0,5.0,6.2,2.7,3.5,3.1,4.3,2.5,4.5,3.4,6.7,3.6,1.6,3.4,1.,4.3])

    video_list = []
    numf_list = []
    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        actidx = database[videoname]['label']

        istrain = True
        step = ratios[actidx]
        if videoname not in trainvideos:
            istrain = False
            step = ratios[actidx]*2

        if fulltest:
            step = 1

        annotations = database[videoname]['annotations']
        numf = database[videoname]['numf']
        numf_list.append(numf)
        num_tubes = len(annotations)
        offset = 4
        frame_labels = np.zeros(numf,dtype=np.int8) # check for each tube if present in
        tube_labels = np.zeros((numf,num_tubes),dtype=np.int16) # check for each tube if present in
        tube_boxes = [[[] for _ in range(num_tubes)] for _ in range(numf)]
        for tubeid, tube in enumerate(annotations):
            # print('numf00', numf, tube['sf'], tube['ef'])
            frame_labels[max(0, tube['sf']-offset):min(numf, tube['ef']+offset)] = 1
            for frame_id, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)): # start of the tube to end frame of the tube

                label = tube['label']
                assert actidx == label, 'Tube label and video label should be same'
                box = tube['boxes'][frame_id, :]  # get the box as an array
                box = box.astype(np.float32)
                box -= 1  # make it 0 230 and 0 to 319
                box[2] += box[0]  #convert width to xmax
                box[3] += box[1]  #converst height to ymax
                box[0] /= 320.0
                box[1] /= 240.0
                box[2] /= 320.0
                box[3] /= 240.0
                tube_labels[frame_num, tubeid] = 1 # change label in tube_labels matrix to 1 form 0
                tube_boxes[frame_num][tubeid] = box  # put the box in matrix of lists

        possible_frame_nums = np.arange(0, numf-0.5, step)
        # print('numf',numf,possible_frame_nums[-1])
        for frame_id, frame_num in enumerate(possible_frame_nums): # loop from start to last possible frame which can make a legit sequence
            frame_num = np.int(frame_num)
            check_tubes = tube_labels[frame_num,:]

            if np.sum(check_tubes)>0:  # check if there aren't any semi overlapping tubes
                all_boxes = []
                labels = []
                image_name = imagesDir + videoname+'/{:05d}.jpg'.format(frame_num+1)
                assert os.path.isfile(image_name), 'Image does not exist'+image_name
                for tubeid, tube in enumerate(annotations):
                    label = tube['label']
                    if tube_labels[frame_num, tubeid] > 0:
                        box = np.hstack((tube_boxes[frame_num][tubeid], label))
                        all_boxes.append(box)
                        labels.append(label)
                if istrain: # if it is training video
                    trainlist.append([vid, frame_num+1, all_boxes])
                    train_action_counts[actidx] += len(labels)
                else: # if test video and has micro-tubes with GT
                    testlist.append([vid, frame_num+1, all_boxes])
                    test_action_counts[actidx] += len(labels)
            elif fulltest or (use_bg and frame_labels[frame_num]==0): # if test video with no ground truth and fulltest is trues
                if istrain:
                    trainlist.append([vid, frame_num + 1, [np.asarray([0., 0., 1., 1., 9999])]])
                    train_action_counts[actidx] += 1  # len(labels)
                else:
                    testlist.append([vid, frame_num + 1, [np.asarray([0., 0., 1., 1., 9999])]])
                    test_action_counts[actidx] += 1  # len(labels)
    print_str = ''
    for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
        tmp_str = 'train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx] , int(actidx), CLASSES['ucf24'][actidx])
        print(tmp_str)
        print_str += tmp_str+'\n'

    tmp_str = 'Trainlistlen ' + str(len(trainlist)) + ' testlist ' + str(len(testlist))
    print(tmp_str)
    print_str += tmp_str + '\n'

    return trainlist, testlist, video_list, numf_list, print_str

def make_lists_daly(rootpath, bg_step=40, use_bg=False, fulltest=False):

    print('root::{} bg {} fulltest{}'.format(rootpath,use_bg,fulltest))
    with open(rootpath + 'splitfiles/finalAnnots.json','r') as f:
        finalAnnot = json.load(f)
    db = finalAnnot['annots']
    testvideos = finalAnnot['testvideos']
    vids = finalAnnot['vidList']
    # pdb.set_trace()
    train_action_counts = np.zeros(len(CLASSES['daly']), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES['daly']), dtype=np.int32)

    video_list = []
    numf_list = []
    trainlist = []
    testlist = []
    count = 0
    for vid, videoname in enumerate(vids):
        istrain = videoname not in testvideos
        vid_info = db[videoname]
        numf = vid_info['numf']
        numf_list.append(numf)
        video_list.append(videoname)
        tubes = vid_info['annotations']
        keyframes = dict()
        frame_labels = np.zeros(numf, dtype=np.int8)  # check for each tube if present in
        offset = 5
        step = bg_step
        if not istrain:
            step = bg_step*2

        if fulltest:
            step = 1
        for tid, tube in enumerate(tubes):
            frame_labels[max(0, tube['sf'] - offset):min(numf, tube['ef'] + offset)] = 1
            for id, fn in enumerate(tube['frames']):
                count += 1
                if not str(fn) in keyframes.keys():
                    keyframes[str(fn)] = {'boxes':[np.hstack((tube['bboxes'][id], tube['class']))]}
                else:
                    keyframes[str(fn)]['boxes'].append(np.hstack((tube['bboxes'][id], tube['class'])))


        possible_frames = [fn for fn in range(0, numf, step)]
        for fn in keyframes.keys():
            if int(fn) not in possible_frames:
                possible_frames.append(int(fn))

        for fn in possible_frames:
            gt_frame = False
            if str(fn) in keyframes.keys():
                boxes = keyframes[str(fn)]['boxes']
                gt_frame = True
            else:
                boxes = [np.asarray([0., 0., 1., 1., 9999])]

            if gt_frame:
                if istrain:
                    trainlist.append([vid, fn + 1, boxes])
                    for box in boxes:
                        label = box[-1]
                        train_action_counts[int(label)] += 1  # len(labels)
                else:
                    testlist.append([vid, fn + 1, boxes])
                    for box in boxes:
                        label = box[-1]
                        test_action_counts[int(label)] += 1  # len(labels)
            elif fulltest or (use_bg and frame_labels[fn] == 0):
                if istrain:
                    trainlist.append([vid, fn + 1, boxes])
                else:
                    testlist.append([vid, fn + 1, boxes])

    print_str = ''
    for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
        tmp_str = 'train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx] , int(actidx), CLASSES['daly'][actidx])
        print(tmp_str)
        print_str += tmp_str + '\n'

    tmp_str = 'Trainlistlen {} train count {} testlist {} test count {} \n total keyframes with labels {}'.format(len(trainlist),
                np.sum(train_action_counts), len(testlist), np.sum(test_action_counts), count)

    print(tmp_str)
    print_str += tmp_str + '\n'

    return trainlist, testlist, video_list, numf_list, print_str


def get_bboxes(label_file):
    fid = open(label_file, 'r')
    boxes = []
    for line in fid.readlines():
        line = line.rstrip('\n')
        line = line.split(' ')
        box = np.asarray([float(l) for l in line])
        boxes.append(box)
    return boxes


def get_annots(rootpath, frame_list, dataset):
    imglist  = []
    mystr = dataset + ' has '
    objcout = 0
    lc = 0
    for line in frame_list:
        lc += 1
        line = line.rstrip('\r')
        imageid = line.rstrip('\n')
        if dataset == 'voc':
            label_file = rootpath + imageid.replace('JPEGImages', 'labels') + '.txt'
        else:
            label_file = rootpath + 'labels/'+ imageid.split('/')[1] + '.txt'
        if os.path.isfile(label_file):
            bboxes = get_bboxes(label_file)
        else:
            bboxes = [np.asarray([0. ,0. ,0. ,0., 9999])]
        objcout += len(bboxes)
        imglist.append([imageid, bboxes])

    mystr += str(lc)+' images and '+str(objcout) + ' object instances\n'

    return imglist, mystr


def make_object_lists(rootpath, dataset='voc'):
    trainlist_names = readsplitfile(rootpath + 'train.txt')
    testlist_names = readsplitfile(rootpath + 'test.txt')
    print_str = ''
    trainlist, mystr = get_annots(rootpath,trainlist_names,dataset)
    print_str += mystr
    testlist, mystr= get_annots(rootpath,testlist_names,dataset)
    print_str += mystr

    return trainlist, testlist, print_str


def get_coco_classes(rootpath):
    anno_file = rootpath + 'instances_val2017.json'
    with open(anno_file, 'r') as f:
        obj = json.load(f)
    cls_dict = obj['categories']
    id_label = {}
    cls_list = []
    count = 0
    for c in cls_dict:
        cls_list.append(c['name'])
        id = c['id']
        id_label[str(id)] = count
        count += 1

    return cls_dict, cls_list, id_label


def change_id_to_labels(trainlist, id_label):
    img_list =[]
    for img in trainlist:
        new_box = []
        for box in img[1]:
            if box[4]<9999:
                new_box.append(np.hstack((box[:4], id_label[str(int(box[4]))])))
            else:
                new_box.append(box)
        img_list.append([img[0], new_box])
    return img_list


def make_lists(dataset, rootpath, imgtype, split=1, use_bg=False, fulltest=False):
    imgpath = os.path.join(rootpath, imgtype)
    if dataset == 'ucf24':
        ## coming form make_lists_ucf ===>>>
        trainlist, testlist, video_list, numf_list, print_str = make_lists_ucf(rootpath, imgtype, split=split, use_bg=use_bg, fulltest=fulltest)
        return CLASSES[dataset], imgpath, trainlist, testlist, video_list, numf_list, print_str
    elif dataset == 'daly':
        trainlist, testlist, video_list, numf_list, print_str = make_lists_daly(rootpath, use_bg=use_bg, fulltest=fulltest)
        return CLASSES[dataset], imgpath, trainlist, testlist, video_list, numf_list, print_str
    elif dataset == 'voc':
        imgpath = os.path.join(rootpath, '%s.jpg')
        trainlist, testlist, print_str = make_object_lists(rootpath)
        return CLASSES[dataset], imgpath, trainlist, testlist, 0, 0, print_str
    elif dataset == 'coco':
        imgpath = os.path.join(rootpath, '%s.jpg')
        cls_names, cls_list, id_label = get_coco_classes(rootpath)
        trainlist, testlist, print_str = make_object_lists(rootpath, dataset)
        # import pdb
        # pdb.set_trace()
        trainlist = change_id_to_labels(trainlist, id_label)
        testlist = change_id_to_labels(testlist, id_label)
        return cls_list, imgpath, trainlist, testlist, cls_names, 0, print_str


class LoadImage(object):

    def __init__(self, space='BGR'):
        self.space = space

    def __call__(self, path_img):

        return cv2.imread(path_img)


class Detection(data.Dataset):
    """UCF24 Action Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, image_set, transform=None,  full_test=False):

        self.input_type = args.input_type
        self.action  = args.action
        self.dataset = args.dataset
        self.use_bg = args.use_bg
        input_type = args.input_type+'-images'
        self.root = args.data_root + args.dataset + '/'
        self.image_set = image_set
        self.transform = transform
        self._annopath = os.path.join(self.root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(self.root, input_type)
        self.ids = list()
        self.input_frames = args.input_frames
        self.image_loader = LoadImage()
        self.classes, self._imgpath, trainlist, testlist, video_list, numf_list, print_str = make_lists(self.dataset,
                                                            self.root, input_type, split=1, use_bg=self.use_bg, fulltest=full_test)

        self.print_str = print_str
        self.video_list = video_list
        self.numf_list = numf_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        if self.dataset == 'ucf24' or self.dataset == 'daly':
            return self.pull_item(index)
        else:
            return self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        annot_info = self.ids[index]
        num_input_frames = self.input_frames
        imgs = []

        if self.action:
            frame_num = annot_info[1]
            video_id = annot_info[0]
            targets = annot_info[2]
            videoname = self.video_list[video_id]
            numf = self.numf_list[video_id]

            if numf+1 <= frame_num + num_input_frames//2 + 1:
                ef = min(numf+1,frame_num + num_input_frames//2 + 1)
                sf = ef-num_input_frames
            else:
                sf = max(frame_num - num_input_frames//2, 1)
                ef = sf+num_input_frames
            frames_ids = np.arange(sf,ef)

            for fn in frames_ids:
                img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, fn)
                img = self.image_loader(img_name)
                # height, width, channels = img.shape
                imgs.append(img)
        else:
            img_id = annot_info[0]
            targets  =  annot_info[1]
            # print(self._imgpath, img_id)
            # pdb.set_trace()
            # print(self._imgpath, img_id)
            img_name = self._imgpath % img_id
            img = self.image_loader(img_name)
            # print(img_name,img.shape)
            imgs.append(img)

        imgs = np.asarray(imgs)
        targets = np.array(targets)
        imgs, boxes, labels = self.transform(imgs, targets[:, :4], targets[:, 4], num_input_frames)

        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        imgs = imgs[:, :, :, (2, 1, 0)]
        if num_input_frames == 1:
            imgs = imgs[0]
            images = torch.from_numpy(imgs).permute(2, 0, 1)
        else:
            images = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
            images = images.view(-1, images.size(2), images.size(3))
        #print(images.size(),target.shape, height)
        return images, target, index

def detection_collate(batch):

    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
            3) image ids for a given input

    """

    targets = []
    imgs = []
    image_ids = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])

    imgs = torch.stack(imgs, 0)
    #print(imgs.size())
    return imgs, targets, image_ids
