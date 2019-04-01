
## Dataset preparations
We support two datasets, [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MS-COCO](http://cocodataset.org/).

First, install `opencv.` 
Navigate to `prep` directory of this repository.


## VOC
Copy `download_voc.sh` to where you want to store `<data_diretory_path>` the dataset, then

`bash download_voc.sh`

It will download VOC dataset and store on `voc` folder under your data directory.

Next, run `python prep/coco_pre_prep.py --base_dir=<path to download directory>` from root directory ot this repostory.

That should preprocess VOC dataset and put `.json` annotation file in `voc` directory.

## COCO

COCO setup is slightly different from VOC setup, 
create `coco` directory and copy `download_coco.sh` from `prep` directory in `coco` directory. 

Now, `bash download_coco.sh`. It will download and unzip images and annotations.


Next, install pycocotools2.0, you can install it by `conda install -c conda-forge pycocotools`

Similiar to VOC setup, run `python prep/coco_pre_prep.py --base_dir=<path to download directory>` from root directory ot this repostory.

Now, we both `coco` and `voc` annotations are in the same format, and we can use the same data loader.
