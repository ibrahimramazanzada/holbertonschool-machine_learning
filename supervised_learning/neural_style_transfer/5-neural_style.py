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
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Static method that rescales an image so its pixel values are
        between 0 and 1 and its largest side is 512 pixels
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))

        image = image[tf.newaxis, ...]
        resized_image = tf.image.resize(
            image, size=[new_h, new_w], method='bicubic')
        scaled_image = tf.clip_by_value(resized_image / 255.0, 0, 1)
        return scaled_image

    def load_model(self):
        """
        Method that loads the VGG19 model and creates a new model that
        outputs the style and content features
        """
        VGG19_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        VGG19_model.save("VGG19_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            "VGG19_base_model", custom_objects=custom_objects)

        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(inputs=vgg.input,
                                           outputs=model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Static method that calculates the gram matrix of an input layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        n = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / n

    def generate_features(self):
        """
        Method that generates the style and content features
        """
        preprocess_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocess_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        style_outputs = self.model(preprocess_style)
        content_output = self.model(preprocess_content)

        self.gram_style_features = [self.gram_matrix(style_layer)
                                    for style_layer in style_outputs[:-1]]
        self.content_feature = content_output[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Method that calculates the style cost for a single layer
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")

        gram_style_output = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style_output - gram_target))

    def style_cost(self, style_outputs):
        """
        Method that calculates the total style cost
        """
        if (not isinstance(style_outputs, list) or
                not all(isinstance(layer, (tf.Tensor, tf.Variable))
                        and len(layer.shape) == 4 for layer in style_outputs)):
            raise TypeError(
                "style_outputs must be a list of tensors of rank 4")

        style_costs = [self.layer_style_cost(style_output, gram_target)
                       for style_output, gram_target in zip(
                           style_outputs, self.gram_style_features)]
        return tf.add_n(style_costs) / len(self.style_layers)
