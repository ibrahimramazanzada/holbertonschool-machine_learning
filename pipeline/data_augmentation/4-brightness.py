#!/usr/bin/env python3
"""randomly changes the brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """randomly changes the brightness of an image"""
    return tf.image.random_brightness(image, max_delta=max_delta)
