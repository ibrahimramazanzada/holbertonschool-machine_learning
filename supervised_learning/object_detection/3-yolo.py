#!/usr/bin/env python3
"""yolov3 object detection model"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """
    Yolo class that uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor method for the Yolo class
        """

        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Method that processes the outputs of the Yolo model
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size[0], image_size[1]
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i in range(len(outputs)):
            grid_h, grid_w, anchor_boxes, _ = outputs[i].shape

            t_xy = outputs[i][..., :2]
            t_wh = outputs[i][..., 2:4]
            box_confidence = outputs[i][..., 4:5]
            box_class_prob = outputs[i][..., 5:]

            # Grid of cell coordinates
            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            grid = np.stack((grid_x, grid_y), axis=-1)
            grid = np.expand_dims(grid, axis=2)

            # Box center, as a fraction of the input image
            box_xy = (1 / (1 + np.exp(-t_xy))) + grid
            box_xy /= [grid_w, grid_h]

            # Box width/height, as a fraction of the input image
            anchors = self.anchors[i]
            box_wh = anchors * np.exp(t_wh)
            box_wh /= [input_width, input_height]

            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)

            box = np.concatenate((box_x1y1, box_x2y2), axis=-1)
            # Scale to the original image size
            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-box_confidence)))
            box_class_probs.append(1 / (1 + np.exp(-box_class_prob)))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Method that filters the boxes based on their confidence scores
        """
        box_scores = [box_confidences[i] * box_class_probs[i]
                      for i in range(len(boxes))]
        box_classes = [np.argmax(box_scores[i], axis=-1)
                       for i in range(len(boxes))]
        box_class_scores = [np.max(box_scores[i], axis=-1)
                            for i in range(len(boxes))]

        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []

        for i in range(len(boxes)):
            mask = box_class_scores[i] >= self.class_t
            filtered_boxes.append(boxes[i][mask])
            filtered_classes.append(box_classes[i][mask])
            filtered_scores.append(box_class_scores[i][mask])

        if len(filtered_boxes) > 0:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            filtered_classes = np.concatenate(filtered_classes, axis=0)
            filtered_scores = np.concatenate(filtered_scores, axis=0)
        else:
            filtered_boxes = np.array([])
            filtered_classes = np.array([])
            filtered_scores = np.array([])

        return (filtered_boxes, filtered_classes, filtered_scores)

    def non_max_suppression(self, filtered_boxes, filtered_classes,
                            filtered_scores):
        """
        Method that applies non-max suppression to the filtered boxes
        """
        nms_boxes = []
        nms_classes = []
        nms_scores = []

        unique_classes = np.unique(filtered_classes)

        for cls in unique_classes:
            cls_mask = filtered_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = filtered_scores[cls_mask]

            # Sort boxes by scores
            sorted_indices = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            while len(cls_boxes) > 0:
                nms_boxes.append(cls_boxes[0])
                nms_classes.append(cls)
                nms_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                ious = self.iou(cls_boxes[0], cls_boxes[1:])
                keep_indices = np.where(ious < self.nms_t)[0] + 1
                cls_boxes = cls_boxes[keep_indices]
                cls_scores = cls_scores[keep_indices]

        return (np.array(nms_boxes),
                np.array(nms_classes), np.array(nms_scores))
