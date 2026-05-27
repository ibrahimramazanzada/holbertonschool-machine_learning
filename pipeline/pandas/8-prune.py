#!/usr/bin/env python3
'''dropna'''


def prune(df):
    '''returns a pd.DataFrame with all rows with missing values removed'''
    return df['Close'].dropna()
