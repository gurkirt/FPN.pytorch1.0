
from modules.box_utils import point_form, jaccard
from modules.prior_box import PriorBox
import torch
import numpy as np
from data.detectionDatasets import make_lists

base_dir = '/home/gurkirt/'
base_dir = '/mnt/mercury-fast/datasets/'
dataset = 'voc'
use_cuda = False
input_dim = 600
thresh = .5
priorbox = PriorBox(input_dim=input_dim, is_cuda=use_cuda)
priors = priorbox.forward()
num_priors = priors.size(0)

classes, imgpath, trainlist, testlist, video_list, numf_list, print_str = make_lists(dataset=dataset, rootpath=base_dir+dataset+'/', imgtype='rgb', split=1, use_bg=False, fulltest=False)

all_recall = torch.FloatTensor(len(trainlist)*30,1)
count = 0
for index in range(len(trainlist)):
    annot_info = trainlist[index]
    targets = np.asarray(annot_info[1])
    bboxes = torch.FloatTensor(targets[:, :4])
    overlaps = jaccard(bboxes, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # print(torch.sum(best_prior_overlap>thresh))
    for bi in range(best_prior_overlap.size(0)):
        bo = best_prior_overlap[bi]
        all_recall[count, :] = bo
        count += 1
all_recall = all_recall[:count]
print('total recall percentage is ', 100*torch.sum(all_recall>thresh)/count, torch.mean(all_recall))