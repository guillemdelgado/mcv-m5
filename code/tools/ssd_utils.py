import numpy as np
import pickle
import urllib
import os
import tensorflow as tf

import keras.backend as K

from tools.yolo_utils import BoundBox, box_iou, prob_compare

"""
    SSD utilities
    from: https://github.com/rykov8/ssd_keras
"""

"""Some utils for SSD."""


class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.
    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.
    # References
        https://arxiv.org/abs/1512.02325
    """

    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes + 1

        # get default priors (https://github.com/rykov8/ssd_keras/raw/master/prior_boxes_ssd300.pkl)
        if not os.path.isfile("prior_boxes_ssd300.pkl"):
            print('   Downloading SSD priors')
            urllib.urlretrieve("https://github.com/rykov8/ssd_keras/raw/master/prior_boxes_ssd300.pkl",
                               "prior_boxes_ssd300.pkl")

        self.priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
        self.num_priors = 0 if self.priors is None else len(self.priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = K.get_session()

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def ssd_build_gt_batch(self, batch_gt, image_shape):

        # First convert batch_gt to the format required by assign_boxes
        # boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes), num_classes without background

        targets = []

        for i, gt in enumerate(batch_gt):
            n_boxes = gt.shape[0]
            boxes = np.zeros((n_boxes, 4 + self.num_classes - 1))  # -1 to not count background
            for j, box in enumerate(gt):
                coords = box[1:]  # [xcenter, ycenter, width, height]
                # the code expects [xmin, ymin, xmax, ymax]
                coords[0] = box[1] - box[3] / 2
                coords[1] = box[2] - box[4] / 2
                coords[2] = box[1] + box[3] / 2
                coords[3] = box[2] + box[4] / 2
                boxes[j, 0:4] = coords
                one_hot = np.zeros(self.num_classes - 1)  # -1 to not count background
                one_hot[int(box[0])] = 1.
                boxes[j, 4:] = one_hot
            y = self.assign_boxes(boxes)
            targets.append(y)

        return np.array(targets)

    def iou(self, box):
        """Compute intersection over union for the box with all priors.
        # Arguments
            box: Box, numpy tensor of shape (4,).
        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.
        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.
        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """

        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        # we encode variance
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                                  assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.
        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.
        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                           np.arange(assign_num),
                                           :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors.
        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.
        # Return
            decode_bbox: Shifted priors.
        """
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.
        # Arguments
            predictions: Numpy array of predicted values for a specific image.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.
        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :4]
        variances = predictions[:, -4:]
        mbox_priorbox = predictions[:, -8:-4]
        mbox_conf = predictions[:, 4:-8]

        decode_bbox = self.decode_boxes(mbox_loc, mbox_priorbox, variances)

        results = []
        boxes = []

        for c in range(self.num_classes):
            if c == background_label_id:
                continue
            c_confs = mbox_conf[:, c]
            c_confs_m = c_confs > confidence_threshold
            if len(c_confs[c_confs_m]) > 0:
                boxes_to_process = decode_bbox[c_confs_m]
                confs_to_process = c_confs[c_confs_m]
                feed_dict = {self.boxes: boxes_to_process,
                             self.scores: confs_to_process}
                idx = self.sess.run(self.nms, feed_dict=feed_dict)
                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]
                labels = c * np.ones((len(idx), 1))
                c_pred = np.concatenate((labels, confs, good_boxes),
                                        axis=1)
                results.extend(c_pred)

        if len(results) > 0:
            results = np.array(results)
            argsort = np.argsort(results[:, 1])[::-1]
            results = results[argsort]
            results = results[:keep_top_k]

            # Convert the bounding boxes into BoundBox objects (to be compatible with the framework)
            for box in results:
                # Get values from the bounding box: label, confidence and coords
                xmin, ymin, xmax, ymax = box[2:]
                label = int(box[0]) - 1  # Background class is 0, so everything must be shifted 1 position to the left
                confidence = box[1]
                confidences = np.zeros(self.num_classes - 1)  # Without background
                confidences[label] = confidence
                # Create a BoundBox object to hold the information
                bx = BoundBox(self.num_classes)
                bx.w, bx.h = 2*(xmax - xmin), 2*(ymax - ymin)
                bx.x, bx.y = xmin + bx.w / 2, ymin + bx.h / 2
                bx.c = confidence
                bx.probs = confidences
                boxes.append(bx)

            return boxes

        else:
            return []