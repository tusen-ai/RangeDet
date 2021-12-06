# RangeDet in MXNet

**An anchor-free single-stage LiDAR-based 3D object detector purely based on the range view representation.**

This is the official implementation of **RangeDet** (ICCV 2021).

## TO DO

- [x] Create range images on KITTI
- [ ] Add installation instructions
- [ ] Add data augmentation
- [x] Add Pedestrian model setting
- [ ] Add full data training results

## Introduction

**RangeDet:In Defense of Range View for LiDAR-based 3D Object
Detection [[arXiv]](https://arxiv.org/abs/2103.10039) [[CVF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_RangeDet_In_Defense_of_Range_View_for_LiDAR-Based_3D_Object_ICCV_2021_paper.pdf)
[[Supplementary materials]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Fan_RangeDet_In_Defense_ICCV_2021_supplemental.pdf)**

[Lue Fan*](https://lue.fan/), Xuan Xiong*, [Feng Wang](http://happynear.wang/), [Naiyan Wang](https://winsty.net/)
, [Zhaoxiang Zhang](https://zhaoxiangzhang.net/)

*Equal contribution

We propose a pure range-view-based framework - RangeDet, which includes **range conditioned pyramid**, **Meta-Kernel**,
**weighted non-maximum suppression**, and a **iou-awared classification loss** to overcome a couple of challenges in
range-view-based 3D detector and achieves comparable performance with state-of-the-art multi-view-based detectors.

## Quick Overview

```
datasets
    create_range_image_roidb.py
rangedet
    symbol
        backbone
            dla_backbone.py
            meta_kernel.py
        head
            builder.py
            loss.py
    core
        detection_metric.py
        input.py
        util_func.py
config 
    ...
scripts 
     distributed_train_and_test.sh
tools
    train.py
    test.py
    create_prediction_bin_3d.py
```

*Some extra points and reminders：*

* `operator_cxx/contrib` contain MXNet's Custom Op written in C++, which need to be compiled with MXNxt.
* `operator_cxx/src_cxx` contain wnms Op written in C++.
* `pybind_cxx` contain bbox assigner function written in C++.
* `operator_py` contain MXNet's Custom Op written in Python.
* `mxnext` is a foundational library, we have modified some functions in this library.
* `utils` contain modular python files, most of them copy from [SimpleDet](https://github.com/TuSimple/simpledet).

## Models Performance

### Results of vehicle evaluated on WOD validation split

Experiments were performed using uniformly sample 25% training data (∼40k frames)

Method | Train epoch | data size | w/wo aug | Overall LEVEL 1 3D-AP/APH on Vehicle (IoU=0.7) | Overall LEVEL 2 3D-AP/APH on Vehicle (IoU=0.7)
--- | --- | --- | --- | --- | ---
**rangedet_veh_wo_aug_4_18e**  | 18 | 1/4 | no | **67.2/66.6** | **58.6/58.1** |
**rangedet_ped_wo_aug_4_18e**  | 18 | 1/4 | no | **65.0/60.0** | **56.2/51.8** |
**rangedet_veh_wo_aug_all_36e**  | 36 | 1/1 | no | **70.1/69.6** | **62.9/62.4** |
**rangedet_ped_wo_aug_all_36e**  | 36 | 1/1 | no | **70.9/66.4** | **61.8/57.8** |

We will further add the experiment results here, for now please refer to our overall performance on Waymo leaderboard in
this [link](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&emailId=5854f8ae-6285&timestamp=1610168529676138)
.

## Installation

Installing and compiling rangedet in the following command.

```bash
pip install -v -e .
```

We will further add requirements and installation instructions on how to install [Horovod](https://horovod.ai/)
, [OpenMPI](https://www.open-mpi.org/) and [MXNet](https://mxnet.apache.org/versions/1.8.0/).

[comment]: <> (## Pretrained Model)

## Training on Waymo Open Dataset

To train on the Waymo Open Dataset:

* Download the Waymo Open Dataset
* Unzip all `*.tar` files and merge them into three directories
* Make sure to put the files as the following structure:

```
datasets
    waymo
        training/*.tfrecord
        validation/*.tfrecord
        testing/*.tfrecord
```

* Next you need to extract the data from tfrecord; We provide the python script in `datasets/create_range_image_roidb.py`
* Store the training data in `npz` format and the label information in `pickle` format respectively

To run the python script `datasets/create_range_image_roidb.py`, do the following settings in the file:

```bash
python datasets/create_range_image_roidb.py --data_path  "**/datasets/waymo" --save_path "**/datasets/waymo-range" --dataset-split 'training' --save_dir 'npz_data'
```

* After that, make sure to put the files as the following structure:

```
datasets
    waymo-range
        training/*.roidb
        validation/*.roidb
        testing/*.roidb
        npz_data/**/*.npz
```

* Note: you need set `DatasetParam.data_root = 'path/to/datasets/waymo-range'` in the `config/*.py` file.

### Single GPU training

Most of the configuration files that we provide assume that we are running on 8 GPUs. In order to be able to run it on
fewer GPUs, there are a few possibilities:

**1. Run the following**

```bash
python tools/train.py --config "/path/to/config/file.py"
```

### Multi-GPU training

```bash
./scripts/horovodrun.sh 8 "python tools/train.py --config /path/to/config/file.py"
```

Note you don't need to change any config between single GPU training and multi-GPU training.

### Mixed precision training

To enable, just do Single-GPU or Multi-GPU training and set `General.fp16 = True` in the `config/*.py` file.

## Create High-quality Range Images on KITTI
We use Hough Transformation to obtain scanning parameters following [RCD](https://arxiv.org/abs/2005.09927). Based on the scanning parameters, it easy to create high-quality range images.

To create range images on KITTI, just simply run:
```
python ./datasets/create_range_image_in_kitti.py --source-dir your_source_kitti_folder --target-dir your_save_directory

```
We assume that the source KITTI data is in MMDetection3D format. In other words, the folder `your_source_kitti_path` is supposed to contains `kitti_infos_trainval.pkl` and `kitti_infos_test.pkl`. It is easy to create these files following the instruction of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/datasets/kitti_det.md).

## Evaluation on Waymo Open Dataset

You can test your model on single gpus. Here is an example:

```bash
python tools/test.py --config "/path/to/config/file.py"
```

You need to create a bin file so that you can use the official waymo evaluation metrics code to get the evaluation
results.

```bash
python tools/create_prediction_bin_3d.py --exp_path "/path/to/experiments" --config "config_file" -epoch num_epoch --save_bin_path "/path/to/save_bin_file"
```

## Acknowledgments

This project is based on [SimpleDet](https://github.com/TuSimple/simpledet). Thanks Yuntao Chen and his colleagues for
their great work!

## Citations

Please consider citing our paper in your publications if it helps your research.

```
@InProceedings{Fan_2021_ICCV,
    author    = {Fan, Lue and Xiong, Xuan and Wang, Feng and Wang, Naiyan and Zhang, ZhaoXiang},
    title     = {RangeDet: In Defense of Range View for LiDAR-Based 3D Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
}
```

## License

RangeDet is released under the Apache license. See [LICENSE](LICENSE) for additional details.
