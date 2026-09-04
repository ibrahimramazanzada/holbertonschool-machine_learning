#!/usr/bin/env python3
"""Defines the Dataset class for machine translation preprocessing."""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps the pt->en translation dataset for training."""

    def __init__(self):
        """Initialize train/validation splits and sub-word tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for the Portuguese/English dataset.
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en
