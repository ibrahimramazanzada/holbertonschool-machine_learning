#!/usr/bin/env python3
'''creates an array from a pd.DataFrame'''
import pandas as pd


def array(df):
    '''returns a np.ndarray from a pd.DataFrame'''
    df2 = df.loc[:, ['High', 'Close']]
    df2 = df2.iloc[-10:]
    return df2.to_numpy()
