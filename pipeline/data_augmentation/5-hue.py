#!/usr/bin/env python3
"""changes the hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """changes the hue of an image"""
    return tf.image.random_hue(image, max_delta=delta)
