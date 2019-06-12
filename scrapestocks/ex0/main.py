import sys,os
print(sys.version)
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt

style.use('fivethirtyeight')

start=dt.datetime(2000,1,1)
end=dt.datetime(2016,12,31)


symbol = 'TSLA'
server = 'yahoo'
df=web.DataReader(symbol,server,start,end)

# Rather than reading data from Yahoo's finance API to a DataFrame, we can also read data from a CSV file into a DataFrame:
df.to_csv('TSLA.csv')
# Now, we can graph with:
df = pd.read_csv('TSLA.csv', parse_dates=True, index_col=0)


print(df.head())
print(df.columns)
print(df[['High','Low']])


plt.figure()
df['Adj Close'].plot() 
df['High'].plot() 
df['Low'].plot() 
plt.legend()
plt.show();


df['100ma'] = df['Adj Close'].rolling(window=100).mean()

plt.figure()
df['100ma'].plot()
df['Adj Close'].plot() 
df['High'].plot() 
df['Low'].plot() 
plt.legend()
plt.show();