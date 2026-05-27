#!/usr/bin/env python3
"""reviewing the data with a plot"""
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('coin.csv')
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

df.loc['2017':]
print(df)


agg_dict = {
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    }
df_resampled = df.resample('D').agg(agg_dict)
    

df_resampled.plot(figsize=(12, 6))
plt.title('Crypto Metrics From 2017 Onward (Daily Aggregation)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()
