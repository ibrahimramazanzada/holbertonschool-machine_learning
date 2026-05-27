#!/usr/bin/env python3
"""swapping levels of a hierarchical index"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """returns a pd.DataFrame with the hierarchical index swapped"""
    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')

    df2 = df2.loc[1417411980:1417417980]
    df1 = df1.loc[1417411980:1417417980]

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])


    df = df.swaplevel(0, 1)
    df = df.sort_index()

    return df
