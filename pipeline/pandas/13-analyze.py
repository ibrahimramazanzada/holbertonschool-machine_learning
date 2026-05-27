#!/usr/bin/env python3
'''df.describe()'''


def analyze(df):
    '''returns a pd.DataFrame with the result of df.describe()'''
    df.drop(columns=['Timestamp'], inplace=True)
    return df.describe()
