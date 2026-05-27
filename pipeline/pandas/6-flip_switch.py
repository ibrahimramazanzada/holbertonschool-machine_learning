#!/usr/bin/env python3
'''sorting and transposing a pd.DataFrame'''


def flip_switch(df):
    '''returns a sorted and transposed pd.DataFrame'''
    df2 = df.sort_values(by='Timestamp', ascending=False)
    return df2.transpose()
