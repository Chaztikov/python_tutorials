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



# If you want to know more about subplot2grid, check out this subplots with Matplotlib tutorial.
# Basically, we're saying we want to create two subplots, and both subplots are going to act like they're on a 6x1 grid, where we have 6 rows and 1 column. The first subplot starts at (0,0) on that grid, spans 5 rows, and spans 1 column. The next axis is also on a 6x1 grid, but it starts at (5,0), spans 1 row, and 1 column. The 2nd axis also has the sharex=ax1, which means that ax2 will always align its x axis with whatever ax1's is, and visa-versa. Now we just make our plots:

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])
plt.show()