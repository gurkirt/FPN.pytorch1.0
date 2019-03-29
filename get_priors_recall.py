
from modules.box_utils import point_form, jaccard
from modules.prior_box_base import PriorBox
import torch
import numpy as np
from data.detectionDatasets import make_object_lists

base_dir = '/home/gurkirt/datasets/'
# base_dir = '/mnt/mercury-fast/datasets/'

input_dim = 300
thresh = 0.5

def main():
    for dataset in ['voc', 'coco']:
        for scales in [[1., 1.33, 1.7], [1., 1.5], [1.]]:
                if dataset == 'coco':
                        train_sets = ['train2017']
                        val_sets = ['val2017']
                else:
                        train_sets = ['train2007', 'val2007', 'train2012', 'val2012']
                        val_sets = ['test2007']

                # classes, trainlist, print_str = make_lists(dataset=dataset, rootpath=base_dir+dataset+'/')
                classes, trainlist, print_str = make_object_lists(base_dir+dataset+'/', train_sets)
                # print(print_str)
                priorbox = PriorBox(input_dim=input_dim, scale_ratios=scales)
                
                priors = priorbox.forward()
                num_priors = priors.size(0)
                print(priors.size()) 
                all_recall = torch.FloatTensor(len(trainlist)*30,1)
                count = 0
                for index in range(len(trainlist)):
                        annot_info = trainlist[index]
                        img_id = annot_info[1]
                        targets = np.asarray(annot_info[3])
                        bboxes = torch.FloatTensor(annot_info[2])
                        overlaps = jaccard(bboxes, point_form(priors))
                        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
                        # print(torch.sum(best_prior_overlap>thresh))
                        for bi in range(best_prior_overlap.size(0)):
                                bo = best_prior_overlap[bi]
                                # print(bo)
                                all_recall[count, :] = bo
                                count += 1
                all_recall = all_recall[:count]
                
                print(scales)
                print('{:s} recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))

def just_whs():
    for dataset in ['voc', 'coco']:
        for scales in [[1., 1.35, 1.75], [1., 1.5], [1.]]:
                if dataset == 'coco':
                        train_sets = ['train2017']
                        val_sets = ['val2017']
                else:
                        train_sets = ['train2007', 'val2007', 'train2012', 'val2012']
                        val_sets = ['test2007']

                # classes, trainlist, print_str = make_lists(dataset=dataset, rootpath=base_dir+dataset+'/')
                classes, trainlist, print_str = make_object_lists(base_dir+dataset+'/', train_sets)
                # print(print_str)
                priorbox = PriorBox(input_dim=input_dim, scale_ratios=scales)
                priors = priorbox.forward()
                print(priors.size())
                unique_priors = priors.numpy()
                unique_priors[:,0] = unique_priors[:,0]*0
                unique_priors[:,1] = unique_priors[:,1]*0
                priors = np.unique(unique_priors, axis=0)
                priors = torch.from_numpy(priors)
                # print(priors) 
                all_recall = torch.FloatTensor(len(trainlist)*30,1)
                count = 0
                for index in range(len(trainlist)):
                        annot_info = trainlist[index]
                        img_id = annot_info[1]
                        targets = np.asarray(annot_info[3])
                        bboxes = torch.FloatTensor(annot_info[2])
                        # print(bboxes)
                        bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
                        bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
                        bboxes[:,0] = bboxes[:,0] * 0.0
                        bboxes[:,1] = bboxes[:,1] * 0.0
                        # print(bboxes)
                        overlaps = jaccard(bboxes, priors)
                        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
                        # print(torch.sum(best_prior_overlap>thresh))
                        for bi in range(best_prior_overlap.size(0)):
                                bo = best_prior_overlap[bi]
                                # print(bo)
                                all_recall[count, :] = bo
                                count += 1
                all_recall = all_recall[:count]
                
                print(scales)
                print('{:s} recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))

if __name__ == '__main__':
#     main()
    just_whs()