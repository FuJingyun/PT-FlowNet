PT-FlowNet
===
This repository contains the PyTorch implementation for PT-FlowNet

## Note
Feel free to open issues and thank you for your attention!


## Installation

### Prerequisites

- CUDA 11.3
- Python 3.8
- PyTorch 1.10
- torch-scatter, h5py, pyyaml, tqdm, tensorboard, scipy, imageio, png

+ Install pointops lib:
```
cd lib/pointops
python3 setup.py install
cd ../..
```

## Usage

### Data Preparation
We follow [HPLFlowNet](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-HPLFlowNet.pdf) to prepare the datasets:
* FlyingThings3D:
Download and unzip the "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" for DispNet/FlowNet2.0 dataset subsets from the [FlyingThings3D website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we used the paths from [this file](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_all_download_paths.txt), now they added torrent downloads)
. They will be upzipped into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

* KITTI Scene Flow 2015
Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
Run the following script for 3D reconstruction:

```bash
python data_preprocess/process_kitti.py --raw_data_path=RAW_DATA_PATH --save_path=SAVE_PATH/KITTI_processed_occ_final --calib_path=util/calib_cam_to_cam
```

## Train
```Shell
sh train.sh
```
`ft3d_dataset_dir` and `kitti_dataset_dir` are the preprocessed dataset paths, please specify the storage location of the dataset on disk. Relevant data file paths can also be predefined in `./tools/parser.py`.
`exp_path` is the experiment folder name and `root` is the project root path. The memory requirement for network training is at least 9573 MiB.

## Train_refine
```Shell
sh train_refine.sh
```
Please add the path of the pre-trained model through `--weights` parameter. 


## Test
Afer training, you can test the model on both `FT3D`and `KITTI` datasets as follows:
```Shell
sh test.sh
```
## Test_refine
```Shell
sh test_refine.sh
```
`--weights` is the absolute path of checkpoint file. The memory requirement for inference is 5249 MiB.

## Acknowledgement
Our code is based on [PV-RAFT](https://github.com/weiyithu/PV-RAFT) and [point-transformer](https://github.com/POSTECH-CVLab/point-transformer). We also refer to [FLOT](https://github.com/valeoai/FLOT) and [HPLFlowNet](https://github.com/laoreja/HPLFlowNet). 
