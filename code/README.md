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


             
