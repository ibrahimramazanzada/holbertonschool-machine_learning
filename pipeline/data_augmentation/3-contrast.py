#!/usr/bin/env python3
"""randomly adjusts the contrast of an image"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """randomly adjusts the contrast of an image"""
    return tf.image.random_contrast(image, lower=lower, upper=upper)
