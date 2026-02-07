# U-Net for Gland Detection in Pathomorphological Images

This repository contains a specialized implementation of the U-Net architecture, adapted specifically for the semantic segmentation of glands in pathomorphological histology images.

## Overview

Accurate segmentation of glandular structures is a critical step in automated cancer grading and diagnosis. This project provides a streamlined, deep learning-based solution using the U-Net architecture, optimizing the original design for the specific textures and shapes found in histopathological tissue samples.

## Architecture

The model utilizes the classic U-Net "encoder-decoder" structure with skip connections to preserve spatial information, allowing for precise localization of gland boundaries.

![U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
*Figure 1: U-Net architecture (Ronneberger et al., 2015).*

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib (for visualization)

## Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/Szymon-Stasiak/U-Net-neural-network.git](https://github.com/Szymon-Stasiak/U-Net-neural-network.git)
cd U-Net-neural-network
pip install -r requirements.txt
```
## Usage
The repository is focused purely on the model implementation. You can integrate the network into your training pipeline as follows:



```python
from unet_core.unet_interface import UNet
from PIL import Image

# 1. Initialize Model
model = UNet(img_height=64, img_width=64)
model.set_model(in_channels=3, out_channels=1, name="path/to/checkpoint")

# 2. Inference
image = Image.open("sample_patch.png")  # Input image (PIL format)
contours = model.find_points(image)     # Returns list of contours

print(f"Detected {len(contours)} gland structures.")

```

### Reference
Original Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation - Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
https://arxiv.org/abs/1505.04597
