#!/usr/bin/env python3
"""crops a size randomly from an image"""
import tensorflow as tf


def crop_image(image, size):
    """crops an image to a given size"""
    return tf.image.random_crop(image, size)
