# Instance Segmentation
This repository is the project 3 for NCTU course IOC5008: Instance Segmentation.

0. [Introduction](#Introduction)
1. [How To Use](#How-To-Use)
2. [Result](#Result)
3. [Reference](#Reference)

## Introduction
The purpose of this project is to train an instance segmentation model on tiny PASCAL VOC dataset which contains only 1349 training images.

## How To Use
I used mask-RCNN provided by [detectron2](https://detectron2.readthedocs.io/). To reproduce this work, you need to follow the [installation guide](https://github.com/facebookresearch/detectron2) to build the environment first.

0. Install environment by following the [guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
1. Download dataset from [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK) and unzip them. The path to the dataset should be the same as the path set in the [code](https://github.com/yamiefun/Instance-Segmentation/blob/main/hw3.py#L20).
2. Train an `Mask-RCNN` with an ImageNet-pretrained `ResNet-50` as its backbone by simply running 
    ```
    python3 hw3.py --mode train --iter [# of training iteration]
    ```
3. Test the trained model and generate an output file called `submission.json` by 
    ```
    python3 hw3.py --mode test
    ```

## Result
In my experiments, I trained 50k iterations and get mAP@0.5: `0.49298`.
## Reference
+ [detectron2](https://github.com/facebookresearch/detectron2)