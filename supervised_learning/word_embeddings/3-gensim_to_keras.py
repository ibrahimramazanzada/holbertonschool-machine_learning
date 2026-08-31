#!/usr/bin/env python3
"""Gensim to keras"""


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a trainable keras Embedding layer.
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
