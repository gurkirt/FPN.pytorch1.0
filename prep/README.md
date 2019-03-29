
## Dataset preprations
We support two datasets, [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MS-COCO](http://cocodataset.org/).

Navigate to `prep' directory of this repository.

## VOC
Copy `download_voc.sh' to where you want to store '<data_diretory_path>' the dataset, then

`bash download_voc.sh'

It will download VOC dataset and store on `voc' folder under your data directory

Next, in order to run `python voc_pre_prep.py' from  'prep' directory chnage `base_dir = '/home/gurkirt/datasets/voc/'' in line 15 to your dataset directory.

That should preprocess VOC dataset and put `.json' annotation file in `voc' directory.