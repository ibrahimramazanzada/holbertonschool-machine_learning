#!/usr/bin/env python3
'''df creation from numpy array'''
import numpy as np
import pandas as pd


def from_numpy(array):
    '''creates a pd.DataFrame from a np.ndarray'''
    return pd.DataFrame(array, columns=[f'{chr(65 + i)}'
                                        for i in range(array.shape[1])])
