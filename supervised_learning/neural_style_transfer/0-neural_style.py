#!/usr/bin/env python3
"""neural style transfer"""
import tensorflow as tf
import numpy as np

class NST:
    """
    Neural Style Transfer class
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor method for the NST class
        """
        if (not isinstance(style_image, np.ndarray) or
           style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError("style_image must be a numpy.ndarray "
                            "of shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
           content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError("content_image must be a numpy.ndarray of"
                            " shape (h, w, 3)")
        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.model = None
        self.style_features = None
        self.content_features = None

    def scale_image(self, image):
        """
        Method that scales an image
        """
        h, w, _ = image.shape
        if h < 256 or w < 256:
            raise ValueError("image height and width must be at least 256")
        # Resize the image to have its largest side equal to 512 pixels
        if h > w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        resized_image = tf.image.resize(
            image, [new_h, new_w], method='bicubic')
        # Scale pixel values to [0, 1]
        scaled_image = resized_image / 255.0
        return scaled_image
