#!/usr/bin/env python3
'''renaming columns and changing data types'''
import pandas as pd


def rename(df):
    '''renames columns and changes data types'''
    df = df.rename(columns={'Timestamp': 'Datetime',})
    df['Datetime'] = df['Datetime'].astype('datetime64[ns]')
    return df[['Datetime','Close']].copy()
