#!/usr/bin/env python3
"""reviewing the data with a plot"""

df.drop(columns=['Weighted_Price'], inplace=True)
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df.set_index('Date', inplace=True)
df['Date'] = df['Date'].astype('datetime64[s]')
df['Close'].ffill(inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)
df.loc[2017-01-01:]
print(df)
