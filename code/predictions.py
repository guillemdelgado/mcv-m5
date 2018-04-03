from __future__ import print_function, division

import os
import numpy as np
import argparse
import glob
import imp

import keras.backend as K
from keras.preprocessing import image

from skimage import data

import scipy.misc

from tools.save_images import my_label2rgboverlay, norm_01
from models.fcn8 import build_fcn8
from models.segnet import build_segnet



import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.ioff()

""" MAIN SCRIPT """

if __name__ == '__main__':

    """ CONSTANTS """
    available_models = {
        'fcn8': build_fcn8,
        'segnet': build_segnet,
    }
    available_datasets = ['camvid', 'cityscapes']
    home_dir = os.path.expanduser('~')
    datasets_dir = os.path.join('/data', 'module5', 'Datasets', 'segmentation')
    dim_ordering = K.image_dim_ordering()
    channel_axis = 3 if dim_ordering == 'tf' else 1
    chunk_size = 16

    """ PARSE ARGS """
    arguments_parser = argparse.ArgumentParser()

    arguments_parser.add_argument('model', help='Model name', choices=available_models)
    arguments_parser.add_argument('dataset', help='Name of the dataset', choices=available_datasets)
    arguments_parser.add_argument('weights', help='Path to the weights file')
    arguments_parser.add_argument('test', help='Path to the folder with the images to be tested')
    arguments_parser.add_argument('--width', help='Target width for the images to be predicted.', type=int)
    arguments_parser.add_argument('--height', help='Target height for the images to be predicted.', type=int)
    arguments = arguments_parser.parse_args()

    model_name = arguments.model
    dataset = arguments.dataset
    weights_path = arguments.weights
    test_dir = arguments.test
    width = arguments.width
    height = arguments.height

    # Create directory to store predictions
    predictions_folder = os.path.join(home_dir, 'prediction-{}-{}'.format(
        model_name,
        dataset
    ))
    try:
        os.makedirs(predictions_folder)
    except OSError:
        # Ignore
        pass

    # Load dataset configuration file
    dataset_config_path = os.path.join(datasets_dir, dataset, 'config.py')

    print()
    print('DATASET CONFIG FILE: {}'.format(dataset_config_path))

    dataset_conf = imp.load_source('config', dataset_config_path)
    classes = dataset_conf.classes
    n_classes = dataset_conf.n_classes
    color_map = dataset_conf.color_map
    void_label = dataset_conf.void_class[0]

    # Images to be predicted
    test_images = glob.glob(os.path.join(test_dir, '*.png'))
    test_images += glob.glob(os.path.join(test_dir, '*.jpg'))
    total_images = len(test_images)
    if total_images == 0:
        print("ERR: path_to_images does not contain any jpg file")
        exit(1)
    print('TOTAL NUMBER OF IMAGES TO PREDICT: {}'.format(total_images))

    # Input shape (get it from first image)
    sample_image_path = test_images[0]
    sample_image = data.load(sample_image_path)
    input_shape = sample_image.shape

    # Target input shape
    target_width = input_shape[0]
    target_height = input_shape[1]
    channels = input_shape[2]

    if dim_ordering == 'th':
        img_shape = (channels, target_height, target_width)
    else:
        img_shape = (target_height, target_width, channels)

    print('TARGET IMAGE SHAPE: {}'.format(img_shape))
    print()

    # Create the model
    model = available_models[model_name](img_shape=img_shape, nclasses=n_classes)

    # Load weights
    model.load_weights(weights_path)

    iteration = 1
    for i in range(0, total_images, chunk_size):
        print()
        print('{:^40}'.format('CHUNK {}'.format(iteration)))

        chunked_img_list = test_images[i:i + chunk_size]
        images = np.array(
            [image.img_to_array(image.load_img(f, target_size=(target_height, target_width))) / 255.
             for f in chunked_img_list]
        )
        num_images_chunk = images.shape[0]

        pred = model.predict(images, batch_size=2, verbose=True)
        print(pred.shape)
        y_pred = np.argmax(pred, axis=2)

        # Void class
        y_pred[(y_pred == void_label).nonzero()] = void_label

        # Store the predictions
        for ind in range(num_images_chunk):
            original_img, label_mask = images[ind], np.reshape(y_pred[ind], (target_height, target_width))
            print(label_mask.shape)
            print(original_img.shape)
	    #if dim_ordering == 'th':
            #    original_img = original_img.transpose((1, 2, 0))

            # img = norm_01(img, mask_batch[j], void_label)*255
            img = norm_01(original_img, label_mask, -1) * 255
            label_overlay = my_label2rgboverlay(label_mask, colors=color_map,
                                                image=img, bglabel=void_label,
                                                alpha=0.6)
            out_name = os.path.join(predictions_folder, os.path.basename(chunked_img_list[ind]))
            scipy.misc.toimage(label_overlay).save(out_name)
            print('Predicted and saved {} ({} / {})'.format(out_name, ind + 1, num_images_chunk))

        iteration += 1
