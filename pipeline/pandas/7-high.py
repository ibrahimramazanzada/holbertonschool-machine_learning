#!/usr/bin/env python3
'''just sorting'''


def high(df):
    '''returns a sorted pd.DataFrame'''
    return df.sort_values(by='High', ascending=False)
