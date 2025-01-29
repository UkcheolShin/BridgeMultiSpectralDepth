# Bridging Spectral-wise and Multi-spectral Depth Estimation via Geometry-guided Contrastive Learning

This is the official pytorch implementation of the paper:

 >Bridging Spectral-wise and Multi-spectral Depth Estimation via Geometry-guided Contrastive Learning
 >
 >[Ukcheol Shin](https://ukcheolshin.github.io/), Kyunghyun Lee, Jean Oh
 >
 >IEEE International Conference on Robotics and Automation (ICRA), 2025
 >
 >[[Paper]()]

### Updates
- 2025.01.29: Release source code & pre-trained weights.

## Usage
### 0. Installation
This codebase was developed and tested with the following packages. 
- OS: Ubuntu 20.04.1 LTS
- CUDA: 11.3
- PyTorch: 1.9.1
- Python: 3.9.16

You can build your conda environment with the following commands.
```bash
conda create python=3.9 pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge --name MS2_bench
conda activate MS2_bench
conda install core iopath pytorch3d -c pytorch -c conda-forge -c pytorch3d -y
pip install mmcv pytorch_lightning timm setuptools==59.5.0 matplotlib imageio path
```

### 1. Dataset
You can download MS2 dataset [here](https://sites.google.com/view/multi-spectral-stereo-dataset/home).
For the train/val/test list for MS2 dataset, copy the txt files from "[MS2dataset Github](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset)"
After downloading the dataset, locate the dataset as follows:
```shell
<datasets>
|-- <KAIST_MS2>
    |-- <sync_data>
    |-- <proj_depth>
    |-- ...
    |-- train_list.txt
    |-- val_list.txt
    |-- ...
```


### 2. Training
1. Download backbone model weights (e.g., swin transformer) if you need. 
For the NeWCRF model training, please download pre-trained Swin-V1 weights [here](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file). 

After downloading the weights, locate them "pt_weights" folder as follows:
```bash
<pt_weights>
|-- <swin_tiny_patch4_window7_224_22k.pth>
|-- ...
|-- <swin_large_patch4_window7_224_22k.pth>
```

2. Train a model with the config file.
If you want to change hyperparamter (e.g., batch size, epoch, learning rate, etc), edit config file in 'configs' folder.

> Single GPU, MS2 dataset, MSDepth_Midas model
```bash
# Stage 1. Align
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/MultiSpectralDepth/MSDepth_midas_stage1.yaml --num_gpus 1 --exp_name MSDepth_MS2_midas_align_singleGPU
# Stage 2. Select
# need to change 'ckpt_path' with the first stage's checkpoint in config files.
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/MultiSpectralDepth/MSDepth_midas_stage2.yaml --num_gpus 1 --exp_name MSDepth_MS2_midas_select_singleGPU
```

> Multi GPUs, MS2 dataset, MSDepth_NewCRF model
```bash
# Stage 1. Align
CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./configs/MultiSpectralDepth/MSDepth_newcrf_stage1.yaml --num_gpus 2 --exp_name MSDepth_MS2_newcrf_align_MS2_multiGPU

# Stage 2. Select
# need to change 'ckpt_path' with the first stage's checkpoint in config files.
CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./configs/MultiSpectralDepth/MSDepth_newcrf_stage1.yaml --num_gpus 2 --exp_name MSDepth_MS2_newcrf_select_MS2_multiGPU
```

3. Start a `tensorboard` session to check training progress. 
```bash
tensorboard --logdir=checkpoints/ --bind_all
```
You can see the progress by opening [https://localhost:6006](https://localhost:6006) on your browser. 

### 3. Evaluation

Evaluate the trained model by running
1. Spectral-wise depth inference
```bash
# MS2-day evaluation set / Monocular depth estimation / modality: rgb, nir, thr
CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --test_env test_day  --save_dir ./results/<target_model>/thr_day --modality <modal>

# MS2-night evaluation set / Monocular depth estimation / modality: rgb, nir, thr
CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --test_env test_night  --save_dir ./results/<target_model>/thr_day --modality  <modal>

# MS2-rainy_day evaluation set / Monocular depth estimation / modality: rgb, nir, thr
CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --test_env test_rain  --save_dir ./results/<target_model>/thr_day --modality  <modal>
```

2. Multi-spectral fused depth inference
```bash
# MS2-day evaluation set / Multispectral depth estimation / modality: rgb, nir, thr
CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --test_env test_day  --save_dir ./results/<target_model>/thr_day --modality <modal>

# MS2-night evaluation set / Multispectral depth estimation / modality: rgb, nir, thr
CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --test_env test_night  --save_dir ./results/<target_model>/thr_day --modality  <modal>

# MS2-rainy_day evaluation set / Multispectral depth estimation / modality: rgb, nir, thr
CUDA_VISIBLE_DEVICES=0 python test_fuse.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --test_env test_rain  --save_dir ./results/<target_model>/thr_day --modality  <modal>
```

### Demo

Inference demo images by running
```bash
# Download the pre-trained weights
bash ./checkpoints/download_pretrained_weights.sh

# Monocular depth estimation (RGB)
CUDA_VISIBLE_DEVICES=0 python inference_depth.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --save_dir ./demo_results/mono/ --modality rgb
# Monocular depth estimation (NIR)
CUDA_VISIBLE_DEVICES=0 python inference_depth.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --save_dir ./demo_results/mono/ --modality nir
# Monocular depth estimation (THR)
CUDA_VISIBLE_DEVICES=0 python inference_depth.py --config ./configs/MultiSpectralDepth/<target_model>.yaml --ckpt_path "PATH for WEIGHT" --save_dir ./demo_results/mono/ --modality thr
```

## Result
We offer the pre-trained model weights that reported on [the paper]().

To reproduce the reported results, follows the below instructions.
```bash
# Download the pre-trained weights
bash ./checkpoints/download_pretrained_weights.sh

# Run scripts for spectral-wise and multi-spectral depth estimation.
bash ./scripts/run_spectral_wise_depth.sh && bash ./scripts/run_multi_spectral_depth.sh
```

### Spectral-wise Monocular Depth Estimation Results
The results are averaged over MS^2 evaluation sets (test_day, test_night, test_rain).

|   Models   | TestSet | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|MiDas|RGB|0.115|0.722|4.269|0.154|0.873|0.976|0.994|
|MiDas|NIR|0.122|0.791|4.099|0.157|0.864|0.973|0.992|
|MiDas|THR|0.088|0.387|3.161|0.120|0.922|0.989|0.998|
|NewCRF|RGB|0.096|0.504|3.657|0.130|0.910|0.987|0.997|
|NewCRF|NIR|0.107|0.571|3.618|0.139|0.896|0.983|0.995|
|NewCRF|THR|0.079|0.310|2.860|0.107|0.940|0.994|0.999|

### Multi-Spectral Depth Estimation Results
|   Models   | TestSet | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|MiDas|RGB|0.107|0.607|3.989|0.143|0.890|0.983|0.996|
|MiDas|NIR|0.110|0.573|3.694|0.144|0.887|0.982|0.996|
|MiDas|THR|0.086|0.382|3.090|0.122|0.928|0.990|0.998|
|NewCRF|RGB|0.087|0.408|3.366|0.119|0.928|0.992|0.998|
|NewCRF|NIR|0.095|0.423|3.255|0.125|0.917|0.990|0.997|
|NewCRF|THR|0.072|0.251|2.623|0.098|0.954|0.996|1.000|

## License
Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Our code is licensed under a [MIT License](LICENSE).

## Citation

Please cite the following paper if you use our work in your research.

```
	@inproceedings{TBA
	}
```

## Related projects & Acknowledgement
Each network architecture built upon the following codebases. 
* [MiDaS](https://github.com/isl-org/MiDaS)
* [NeWCRF](https://github.com/aliyun/NeWCRFs)