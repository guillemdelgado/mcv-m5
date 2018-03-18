# Keras implementation of Classification, Detection and Segmentation Networks

## Introduction

This repo contains the code to train and evaluate state of the art classification, detection and segmentation methods in a unified Keras framework working with Theano and/or TensorFlow. Pretrained models are also supplied.

## Available models and how to run

### Classification
Note: All the configuration files except the Baseline model and Adam optimizer experiment have samplewise mean substraction and std division as input preprocessing to perform normalization.

 - [x] VGG16 and VGG19 network as described in [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf).
 
 * Baseline Model using TT100K_trafficSigns Dataset:
 
       python train.py -c config/tt100k_classif_include_test.py -e vgg_baseline
       
 * Baseline Model using Adam optimizer instead of RMSProp using TT100K_trafficSigns Dataset::

       python train.py -c config/tt100k_classif_adam.py -e vgg_adam_optimizer
       
 * Baseline Model using Data Augmentation in TT100k_trafficSigns Dataset:
  
       python train.py -c config/tt100k_classif_data_augm.py -e vgg_data_augm

       
 * Taking as input crops of (224, 224) dimension using TT100K_trafficSigns Dataset:
 
       python train.py -c config/tt100k_classif_cropping.py -e vgg_cropping
    
 * Transfer Learning to BelgiumTSC dataset using weights from TT100k_trafficSigns Dataset:
         
       python train.py -c config/belgiumtsc_classif_finetuning.py -e vgg_finetuning

 *  Baseline Model trained from scratch using KITTI Dataset:
 
        python train.py -c config/kitti_classif_scratch.py -e vgg_from_scratch
        
 * Baseline Model fine-tuning from ImageNet weights using KITTI Dataset:
 
         python train.py -c config/kitti_classif_finetune_imagenet.py -e vgg_finetune_imagenet

 - [x] ResNet based DNN with Stochastic Depth as described in [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf).
 
 * Deep Network with Stochastic Depth based on Residual Blocks using TT100k_trafficSigns Dataset:
          
          python train.py -c config/tt100k_classif_stochastic_depth.py -e stochastic_depth


### Detection
 - [x] YOLO9000 and Tiny-YOLO as described in [YOLO9000: better, faster, stronger](http://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf).

 * YOLO Model with TT100k Dataset:
 
        python train.py -c config/tt100k_yolo.py -e yolo_baseline
        
 * YOLO Model using Data Augmentation with TT100k Dataset:
 
       python train.py -c config/tt100k_yolo_da.py -e yolo_data_augm
        
 * Tiny-YOLO Model with TT100k Dataset:
 
       python train.py -c config/tt100k_tiny_yolo.py -e tiny_yolo_baseline
 
 * YOLO Model with Udacity Dataset:
 
        python train.py -c config/udacity_yolo.py -e yolo_baseline
        
 - [x] SSD as described in [Ssd: Single shot multibox detector](https://arxiv.org/pdf/1512.02325).     
 
 * SSD Model with TT100k Dataset:
 
        python train.py -c config/tt100k_ssd.py -e ssd_baseline
        
 * SSD Model with Udacity Dataset:
 
         python train.py -c config/udacity_ssd.py -e ssd_baseline
         
             
