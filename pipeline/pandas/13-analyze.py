#!usr/bin/env python3
'''df.describe()'''


def analyze(df):
    '''returns a pd.DataFrame with the result of df.describe()'''
    return df.describe(exclude='Timestamp')
