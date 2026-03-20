<div align="center">

# BD-YOLO  
### Boosting Small Bridge Detection via Edge-Aware Super-Resolution and Gabor–Laplacian Convolutions

[![Paper](https://img.shields.io/badge/Paper-IEEE%20GRSL-blue.svg)](https://doi.org/10.1109/LGRS.2026.3673615)
[![Task](https://img.shields.io/badge/Task-Bridge%20Detection-green.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Remote%20Sensing-orange.svg)]()

</div>

---

## Overview

**BD-YOLO** is a bridge detection framework for remote sensing imagery, designed to improve the recognition of **small bridges** under complex backgrounds.  
The method enhances fine-detail perception through:

- **Edge-Aware Super-Resolution**
- **Gabor–Laplacian Convolutions**

This repository provides resources for **bridge detection** research, including dataset information and citation details for the published paper.

---

## Dataset

This project supports training and evaluation on the following bridge-detection dataset.

### MBDD (Bridge Detection Dataset)

MBDD is a **self-constructed bridge detection dataset** designed for optical remote sensing imagery at **sub-meter spatial resolution**.  
It covers the **Beijing–Tianjin–Hebei region in China** and is built to support bridge detection under diverse geographic and background conditions.

#### Key Features

🌍 **Regional Coverage**  
Collected from the Beijing–Tianjin–Hebei region, with rich scene diversity across different geographic areas.

🖼️ **Fixed Image Size**  
Contains **11,461 optical remote sensing images**, each with a spatial size of **1024×1024 pixels**.

🏗️ **Bridge Annotations**  
A total of **16,069 bridge instances** are manually annotated for detection tasks.


> MBDD is intended to provide a practical benchmark for bridge detection in high-resolution remote sensing imagery, especially in scenes with complex backgrounds and elongated small targets.

#### Download
- **PATREO / UFMG download page**  
  [Google Drive Download Link](https://drive.google.com/file/d/1JYMXYKfcCmLDGkaR1CBmqDRGQgHmwGfs/view?usp=drive_link)


---

## Paper

If you find this work useful in your research, please consider citing our paper:

**BD-YOLO: Boosting Small Bridge Detection via Edge-Aware Super-Resolution and Gabor–Laplacian Convolutions**  
*IEEE Geoscience and Remote Sensing Letters (GRSL), 2026*

- **DOI:** [10.1109/LGRS.2026.3673615](https://doi.org/10.1109/LGRS.2026.3673615)

---

## Citation

```bibtex
@ARTICLE{11432906,
  author={Shi, Chengyi and Li, Xin and Lyu, Xin and Zhang, Xuejie and Li, Xinyu and Gong, Che and Yan, Hao and Liu, Jing and Liu, Fan and Gao, Hongmin},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={BD-YOLO: Boosting Small Bridge Detection via Edge-Aware Super-Resolution and Gabor–Laplacian Convolutions},
  year={2026},
  volume={23},
  number={},
  pages={1-5},
  keywords={Bridges;Kernel;Image edge detection;Feature extraction;YOLO;Superresolution;Interference;Spatial resolution;Detectors;Semantics;Bridge detection (BD);convolutional neural network;object detection;remote sensing image (RSI)},
  doi={10.1109/LGRS.2026.3673615}
}
