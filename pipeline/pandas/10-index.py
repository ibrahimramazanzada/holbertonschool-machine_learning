#!/usr/bin/env python3
'''assigning the index to a column'''


def index(df):
    '''index set to the Timestamp column'''
    return df.set_index('Timestamp')
