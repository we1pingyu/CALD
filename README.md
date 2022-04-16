# Consistency-basd Active Learning for Object Detection

## Introduction
This repo is the official implementation of CALD: [**Consistency-basd Active Learning for Object Detection**](https://arxiv.org/abs/2103.10374), accepted to the Workshop on Learning With Limited Labelled Data for Image and Video Understanding (L3D-IVU).

![detail](detail.jpg)
![results](results.png)
## Requirement
- pytorch>=1.7.1
- torch=0.8.2

(option if you want to get class-wise results of coco)

- mmcv=1.0.4
- pycocotools=2.0.2
- terminaltables=3.1.0
## Quick start
```
python cald_train.py --dataset voc2012 --data-path your_data_path --model faster
``` 
## Citation

If you are interested with our work, please cite this project.

```
@article{yu2021consistency,
  title={Consistency-based active learning for object detection},
  author={Yu, Weiping and Zhu, Sijie and Yang, Taojiannan and Chen, Chen and Liu, Mengyuan},
  journal={arXiv preprint arXiv:2103.10374},
  year={2021}
}
```
