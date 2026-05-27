#!/usr/bin/env python3
'''slicing a pd.DataFrame'''


def slice(df):
    '''returns a slice of a pd.DataFrame'''
    df2 = df.loc[:, ['High', 'Low', 'Close', 'Volume_BTC']]
    return df2.iloc[::60, :]
