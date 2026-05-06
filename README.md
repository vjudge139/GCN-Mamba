# GCN-Mamba

## Introduction

This repository has been evaluated and verified on Linux operating systems. The recommended runtime environment requires Python 3.10, along with PyTorch 2.1.0 and CUDA 12.1.

## Table of Contents

- [Introduction](#Introduction)
- [Table of Contents](#Table-of-Contents)
- [Benchmark](#Benchmark)
- [Dataset Preparation](#Dataset-Preparation)
- [Model Training and Testing](#Model-Training-and-Testing)
- [License](#License)

## Benchmark
<table>
  <thead>
    <tr>
      <th align="center">Type</th>
      <th align="center">Model</th>
      <th align="center">Publication</th>
      <th>GitHub Repository</th>
    </tr>
  </thead>
  <tbody>
    <!-- Type: C (1-3行) -->
    <tr>
      <td align="center" rowspan="3">C</td>
      <td align="center">IFNet</td>
      <td align="center">ISPRS'2020</td>
      <td><a href="https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images">Link</a></td>
    </tr>
    <tr>
      <td align="center">SNUNet</td>
      <td align="center">GRSL'2021</td>
      <td><a href="https://github.com/likyoo/Siam-NestedUNet">Link</a></td>
    </tr>
    <tr>
      <td align="center">HANet</td>
      <td align="center">JSTARS'2023</td>
      <td><a href="https://github.com/ChengxiHAN/HANet-CD">Link</a></td>
    </tr>
    <!-- Type: T (4-8行) -->
    <tr>
      <td align="center" rowspan="5">T</td>
      <td align="center">SwinUnet</td>
      <td align="center">TGRS'2022</td>
      <td><a href="https://github.com/CCRG-XJU/ChangeDetection_SwinSUNet_TGRS2022">Link</a></td>
    </tr>
    <tr>
      <td align="center">BIT</td>
      <td align="center">TGRS'2022</td>
      <td><a href="https://github.com/justchenhao/BIT_CD">Link</a></td>
    </tr>
    <tr>
      <td align="center">ACABFNet</td>
      <td align="center">JSTARS'2022</td>
      <td><a href="https://github.com/SONGLEI-arch/ACABFNet/blob/main/README.md">Link</a></td>
    </tr>
    <tr>
      <td align="center">A2Net-LWGANet</td>
      <td align="center">AAAI'2026</td>
      <td><a href="https://github.com/AeroVILab-AHU/LWGANet/tree/main">Link</a></td>
    </tr>
    <tr>
      <td align="center">CLAFA-LWGANet</td>
      <td align="center">AAAI'2026</td>
      <td><a href="https://github.com/AeroVILab-AHU/LWGANet/tree/main">Link</a></td>
    </tr>
    <!-- Type: G (9-10行) -->
    <tr>
      <td align="center" rowspan="2">G</td>
      <td align="center">CF-GCN</td>
      <td align="center">TGRS'2024</td>
      <td><a href="https://github.com/liucongcharles/CF-GCN">Link</a></td>
    </tr>
    <tr>
      <td align="center">BGSINet</td>
      <td align="center">GRSL'2024</td>
      <td><a href="https://github.com/JackLiu-97/BSINet">Link</a></td>
    </tr>
    <!-- Type: M (11-14行) -->
    <tr>
      <td align="center" rowspan="4">M</td>
      <td align="center">RS-Mamba</td>
      <td align="center">GRSL'2024</td>
      <td><a href="https://github.com/KyanChen/RSMamba">Link</a></td>
    </tr>
    <tr>
      <td align="center">ChangeMamba</td>
      <td align="center">TGRS'2024</td>
      <td><a href="https://github.com/ChenHongruixuan/ChangeMamba">Link</a></td>
    </tr>
    <tr>
      <td align="center">CDMamba</td>
      <td align="center">arxiv'2024</td>
      <td><a href="https://github.com/zmoka-zht/CDMamba">Link</a></td>
    </tr>
    <tr>
      <td align="center">CSSM</td>
      <td align="center">GRSL'2025</td>
      <td><a href="https://github.com/Elman295/CSSM">Link</a></td>
    </tr>
    <!-- Type: VFM (15-16行) -->
    <tr>
      <td align="center" rowspan="2">VFM</td>
      <td align="center">FEAWNet</td>
      <td align="center">arxiv'2025</td>
      <td><a href="https://github.com/SUPERMAN123000/FAEWNet">Link</a></td>
    </tr>
    <tr>
      <td align="center">ChangeDINO</td>
      <td align="center">arxiv'2025</td>
      <td><a href="https://github.com/chingheng0808/ChangeDINO">Link</a></td>
    </tr>
  </tbody>
</table>
## Dataset Preparation


### Remote Sensing Change Detection Dataset

#### WHU-CD Dataset

- Data download link: [WHU-CD Dataset BaiDu](https://pan.baidu.com/s/1D1dffQ4FhGW10gVi2AvFbg?pwd=dp9v) Code: dp9v.


#### LEVIR-CD Dataset 

- Data download link: [LEVIR-CD Dataset BaiDu](https://pan.baidu.com/s/1XQns61C-50o1AlEzSJ002A?pwd=y6mt) Code: y6mt.

#### WCD Dataset 

- Data download link: [WCD Sample Dataset](https://pan.baidu.com/s/1rz46rmbUzd_xvqdTICy5Zg?pwd=tyxk) Code: tyxk.
- *Note: Please be aware that this is merely a sample dataset provided for quick testing and demonstration purposes.*


#### Organization Method

Datasets can be acquired from alternative sources. However, to ensure compatibility, please arrange your directory structure strictly as shown below:

```
${DATASET_ROOT}
├── A
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── B
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── label
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── list
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
```

## Model Training and Testing

All customizable settings and hyperparameters for both training and evaluation phases are located within the `config` directory. 

#### Example of Training on WHU-CD Dataset

```shell
python train.py
```

#### Example of Testing on WHU-CD Dataset

```shell
python test.py
```
## License

This project is licensed under the [Apache 2.0 License](LICENSE).
