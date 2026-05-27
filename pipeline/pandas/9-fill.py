#!/usr/bin/env python3
'''one drop and multiple fillna'''


def fill(df):
    '''drop and fill'''
    df.drop(columns=['Weighted_Price'], inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    df['Volume_(BTC)'].fillna(0, inplace=True)
    df['Volume_(Currency)'].fillna(0, inplace=True)
    return df
