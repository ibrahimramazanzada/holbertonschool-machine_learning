#!/usr/bin/env python3
'''concatenating two pd.DataFrames'''
import pandas as pd


def concat(df1, df2):
    '''returns a concatenated pd.DataFrame'''
    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')
    df2 = df2.loc[:1417411920]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    return df
