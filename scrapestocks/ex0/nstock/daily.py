# https://gist.github.com/stoffprof/719ae8d876d3a5539112e74d292339c9
# Read Yahoo price data
import sys, os, re, pickle, subprocess
import bs4 as bs
import requests
print(sys.version)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.serif']='Times'
mpl.rcParams['font.size']= 6.0
# mpl.rcParams['grid.alpha']= 1.0
# mpl.rcParams['grid.color']= '#b0b0b0'
# mpl.rcParams['grid.linestyle']= '-'
mpl.rcParams['grid.linewidth']= 1.8
mpl.rcParams['lines.linewidth']= 2.5
mpl.rcParams['lines.marker']= 'None'
mpl.rcParams['figure.dpi']= 100.0

import numpy as np
import pandas as pd

import os;import numpy as np;import pandas as pd
import os,sys,re,subprocess
import numpy as np
import scipy
import seaborn as sns

import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import datetime

import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot,make_interp_spline
import scipy.sparse.linalg as spla

class YahooDailyReader():
    
    def __init__(self, symbol=None, start=None, end=None):
        import datetime, time
        self.symbol = symbol
        
        # initialize start/end dates if not provided
        if end is None:
            end = datetime.datetime.today()
        if start is None:
            start = datetime.datetime(2010,1,1)
        
        self.start = start
        self.end = end
        
        # convert dates to unix time strings
        unix_start = int(time.mktime(self.start.timetuple()))
        day_end = self.end.replace(hour=23, minute=59, second=59)
        unix_end = int(time.mktime(day_end.timetuple()))
        
        url = 'https://finance.yahoo.com/quote/{}/history?'
        url += 'period1={}&period2={}'
        url += '&filter=history'
        url += '&interval=1d'
        url += '&frequency=1d'
        self.url = url.format(self.symbol, unix_start, unix_end)
        
    def read(self):
        import requests, re, json
       
        r = requests.get(self.url)
        
        ptrn = r'root\.App\.main = (.*?);\n}\(this\)\);'
        txt = re.search(ptrn, r.text, re.DOTALL).group(1)
        jsn = json.loads(txt)
        df = pd.DataFrame(
                jsn['context']['dispatcher']['stores']
                ['HistoricalPriceStore']['prices']
                )
        df.insert(0, 'symbol', self.symbol)
        df['date'] = pd.to_datetime(df['date'], unit='s').dt.date
        
        # drop rows that aren't prices
        df = df.dropna(subset=['close'])
        
        df = df[['symbol', 'date', 'high', 'low', 'open', 'close', 
                 'volume', 'adjclose']]
        df = df.set_index('symbol')
        return df
    
ydr = YahooDailyReader('IBM')
df = ydr.read()