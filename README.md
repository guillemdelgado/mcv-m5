# Scene Understanding for Autonomous Vehicles
This project aims to put in practice concepts and techniques to develop deep neural networks for being able to compute three main computer vision tasks: object detection, segmentation and recognition. These concepts will be applied in  an specific environent which is autonomous vehicles.

## Authors
### DeepDrivers

| Authors  | Emails | Github |
| ------------- | ------------- | ------------- |
| Guillem Delgado  | guillem.delgado@gmail.com  | [guillemdelgado](https://github.com/guillemdelgado) |
| Francisco Roldan | fran.roldans@gmail.com | [franroldans](https://github.com/franroldans) |
| Jordi Gené | jordigenemola.1@gmail.com  | [Jordi-Gene-Mola](https://github.com/Jordi-Gene-Mola) |


[robertbenavente](https://github.com/robertbenavente) and [lluisgomez](https://github.com/lluisgomez/) as supervisors.


## Final report
The final report details and summarizes the work done weekly in the project can be found [here](https://www.overleaf.com/read/nftwzgfcgmbj). 

## Final Slides
The final slides for the presentations detailing and summaraizing the work done weekly can be found [here](https://docs.google.com/presentation/d/1ZIXaVrfedYnxIWHwTlNyLQVscnTAmDpMa9mK6FXyt20/edit?usp=sharing).

## Datasets Analysis
We have manually inspected the data in which we have work to facilitate the interpretation of the results obtained. Find the data set analysis [here](https://docs.google.com/presentation/d/1qLqqRS4AYZBUM-f01pdIpluVP8X4wnHsjYFz7Sntpek/edit?usp=sharing). 

## Weights
The weights of the different models can be found [here](https://drive.google.com/open?id=1eoo47p8RaoioJT5NxVRFH5offR7vTjBO).
As the size of weights files is huge there are just the most successful experiments for each dataset and network. However, if you feel there are missing the weights of an experiment you are interested in, just open an issue and we will update this Google Drive with your request ASAP. 

## Object recognition - Week 2
### Abstract
In order to choose a good-performing object recognition network for our system, we have tested several CNNs with different architectures. Changing different parameters from *code/config/tt100k_classif.py* we were able to test different datasets and different NN. In addition, we implemented and tested a Deep Network with Stochastic Depth based on Residual Blocks which can be found in *code/models/stochastic_depth.py*
### Code
The framework's code is divided as follows:

  * *callbacks/* : Folder that handles all the different callbacks involved during training.
  * *config/* : Folder that contains all configuration files for the different experiments done and handles them.
  * *initializations/* : Useful tools for weights initialization.
  * *layers/* : Folder that contains layers not present in Keras such as Deconvolution.
  * *metrics/* : Tools for model evaluation
  * *models/* : Folder that handles all the different models involved in the project.
  * *tools/*  : Useful tools to manage deep learning projects.
  
### Results
[Results](https://docs.google.com/presentation/d/1_2VD3MA0VBb7Mtlwvp1Wve2bslxj_1RlP51D4AoQJgk/edit?usp=sharing) of the different experiments.
### How to run the code
See this [README](https://github.com/guillemdelgado/mcv-m5/blob/master/code/README.md) to know how to run the code and run the experiments.
### Goals

1. Testing the framework:
- [x] Analyze the dataset, which the summary can be found the [Datasets Analysis](https://github.com/guillemdelgado/mcv-m5/blob/master/README.md#datasets-analysis) section.
- [x] Calculate the accuracy on train and test sets.
- [x] Evaluate different techniques in the configuration file.
- [x] Transfer learning to another dataset.
- [x] Understand configuration file.
2. Train networks on different datasets:
- [x] VGG model from scratch.
- [x] VGG model fine-tuning with ImageNet weights.
3. Implementing a new Neural Network:
- [x] Integrate the new model into the framework.
- [x] Evaluate the new model on TT100K dataset.
4. Boost performance
- [x] Data Augmentation.
- [x] Data Preprossesing.
- [x] Comparative of optimizers.


## Object detection - Week 3 and 4
### Abstract
For object detection we have considered two single-shot models: You Only Look Once (YOLO) with the smaller model, Tiny-YOLO, and Single-Shot Multibox Detector (SSD). All these models have been trained to detect a variety of traffic signs in the TT100K detection dataset and to detect pedestrians, cars and trucks in the Udacity dataset. Faster-RCNN was tried but it is not included due to difficulties to upgrade it to newest Keras version.
### Code
The contributions done for these weeks are:

  * *layers/ssd_layers.py* : Layers needed for the SSD model
  * *models/ssd.py* : SSD Model.
  * *tools/ssd_utils*  : Utils needed for the SSD Model.
  
### Results
[Results](https://docs.google.com/presentation/d/1ZIXaVrfedYnxIWHwTlNyLQVscnTAmDpMa9mK6FXyt20/edit?usp=sharing) of the different experiments.
### How to run the code
See this [README](https://github.com/guillemdelgado/mcv-m5/blob/master/code/README.md) to know how to run the code and run the experiments.
### Goals

1. YOLOv2 model in TT100k Dataset:
- [x] Analyze the dataset, which the summary can be found the [Datasets Analysis](https://github.com/guillemdelgado/mcv-m5/blob/master/README.md#datasets-analysis) section.
- [x] Calculate the F-score.
2. Summary of references:
- [x] Summary of Yolo and F-RCNN.
3. Implementing a new Neural Network:
- [x] Integrate the new model (SSD) into the framework.
- [x] Evaluate the new model on BOTH datasets.
4. Train the networks on a different dataset:
- [x] Evaluate the YOLO on BOTH datasets.
- [x] Evaluate the SSD on BOTH datasets.
5. Boost performance:
- [x] Data Augmentation.
- [x] Data Preprossesing.
- [x] Comparative of optimizers.

## Object Segmentation - Week 5 and 6
### Abstract
Three different models have been tried during these weeks. Starting by FCN-8, we have explored different regularization methods not tried in previous weeks, such as batch normalization, and trained the model in different datasets (CamVid, Cityscapes and KITTI). Segnet and Unet have been also adapted to Keras 2.0 and tested on CamVid dataset. 
### Code
The contributions done for these weeks are:

  *
  
### Results
[Results](https://docs.google.com/presentation/d/1ZIXaVrfedYnxIWHwTlNyLQVscnTAmDpMa9mK6FXyt20/edit?usp=sharing) of the different experiments.
### How to run the code
See this [README](https://github.com/guillemdelgado/mcv-m5/blob/master/code/README.md) to know how to run the code and run the experiments.
### Goals

1. FCN8 model:
- [x] Analyze the dataset, which the summary can be found the [Datasets Analysis](https://github.com/guillemdelgado/mcv-m5/blob/master/README.md#datasets-analysis) section.
2. Summary of references:
- [x] Summary of FCN and SegNet.
3. Implementing a new Neural Network:
- [x] Integrate the new model (SegNet) into the framework.
- [x] Integrate the new model (unet) into the framework.
4. Train the networks on a different dataset:
- [x] Evaluate FCN8 for Cityscapes and KITTI.
5. Boost performance:
- [x] Comparative of batch sizes.
- [x] Comparative or learning rates.
- [x] Comparative of optimizers.
- [x] Batch normalitzation.
- [x] Data augmentation for unet.
## References
[Summaries](https://www.overleaf.com/read/tgxwrbzqdvst)

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

A. Krizhevsky, “Imagenet classification with deep convolutional neural networks,” in Advances in neural information processing systems, 2012.

G. Huang, “Deep networks with stochastic depth,” in European Conference on Computer Vision, 2016.

Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.

Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." arXiv preprint 1612 (2016).

Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.

Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

