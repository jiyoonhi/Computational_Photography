# Overview

In this project, we develop an image background removal system using machine learning techniques. Image background removal is an essential component of image processing with numerous practical applications in the real world. Here, we have trained an image in-painting model using the Partial Convolution to restore or reconstruct an image by filling in the missing parts with plausible information. We integrated the trained image in-painting model with a pre-trained image segmentation model which removes a particular object of an image.

In this final project report, we present the methodology and results of our image background removal project. We describe in detail the steps involved in training and evaluating the model, as well as the integration of the pre-trained image segmentation model. We also discuss the challenges encountered and the solutions implemented to optimize the system's accuracy and speed. Finally, we provide a comprehensive evaluation of our system's performance.


# Image segmentation model

 - For image segmentation we have used the pre-trained mask RCNN model from https://github.com/open-mmlab/mmdetection.git.
 - The dataset was trained on COCO dataset which can indetify and provide instance segmentations for 80 different categories of objects.
 - The [instance_segmentation](https://github.com/insoochung/bg_obj_remover/blob/main/notebooks/instance_segmentation.ipynb) notebook provides the entire workflow to identify different objects and generates the masks for the selected object.


# Image in-painting model

We implement PConvUNet as our image in-painting module using TF2. We take reference of an open source version of [TF1.0 implementation](https://github.com/MathiasGruber/PConv-Keras).

## Training

### Assets

- [VGG16 weights](https://drive.google.com/open?id=1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0)
  - Place in `pconv_unet/assets/`
- Training data [#1](https://www.kaggle.com/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0), [#2](https://www.kaggle.com/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1)
  - Download and unpack the full data (or a subset) in `pconv_unet/assets/kaggle`
- [PConvUNet weights](https://drive.google.com/file/d/1CUmLCMKqEgbIvyny5HdGWen98S3iOA1w/view?usp=sharing)
  - Place in `pconv_unet/assets/`
  - For the provided checkpoint, we used a subset of imagenet data for a limited number of epochs and the resulting checkpoint limited performance.

### Quickstart

Once the assets are ready perform the following:
```bash
cd pconv_unet
pip install -r requirements.txt
python train.py
```

## Inference

Once you have trained the model, follow [notebooks/inpainting.ipynb](notebooks/inpainting.ipynb) to perform image inpainting.

# Contriutions

- Setup image segmentation
  - Inference: Alekhya Duba
- Setup image inpainting
  - Train: Insoo Chung
  - Inference: Jiyoon Hwang
