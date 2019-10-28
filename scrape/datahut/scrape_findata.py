import bs4
import csv

import requests
from bs4 import BeautifulSoup
import pandas as pd 

# https://blog.datahut.co/scraping-yahoo-finance-data-using-python/

'''
Financial market data is one of the most valuable data in the current time. If analysed correctly, it holds the potential of turning an organisation’s economic issues upside down. Among a few of them, Yahoo finance is one such website which provides free access to this valuable data of stocks and commodities prices. In this blog, we are going to implement a simple web crawler in python which will help us in scraping yahoo finance website. Some of the applications of scraping Yahoo finance data can be forecasting stock prices, predicting market sentiment towards a stock, gaining an investive edge and cryptocurrency trading. Also, the process of generating investment plans can make good use of this data!

Before scraping yahoo finance website, let us first understand more about Yahoo finance Data in the next section. 

What is Yahoo Finance?
Yahoo finance is a business media platform from Yahoo which provides comprehensive offerings in the world of business and investment. It has a plethora of available business information like financial news, data about stock quotes, press releases and financial reports. Whether you are an investor or are just looking for some business news, Yahoo finance is the place to go. The biggest plus of Yahoo finance is that it provides all of this information for free. Hence by scraping Yahoo finance data, you can actually get valuable information at your end and do an analysis of stocks and currencies trends. Moreover, you get real-time information about stock prices along with access to other financial investment/management tools.

Why Scrape Finance websites?
Financial data if extracted and analysed in real time can provide a wealth of information for investments, trading, research and sentiment analysis

Stock trading 
Online trading involves stocks trading via an online platform. Online trading portals facilitate the trading of different financial instruments such as stocks, mutual funds and commodities. In online stock trading, owners of one stock meet different buyers virtually and sell the stocks to buyers. The selling part only happens when a buyer and a seller has negotiated the price of exchange.

Furthermore, these prices are market dependent and are provided by scraping yahoo finance. Moreover, stock trading organisations can leverage yahoo finance data to keep a record of changing stock prices and market trend. This analysis will help financial and investment companies to predict the market and buy/sell stocks for maximum profits.

Sentiment analysis of the market 
Organisations can perform sentiment analysis over the blogs, news, tweets and social media posts in business and financial domains to analyse the market trend. Furthermore, scraping Yahoo finance will help them in collecting data for natural language processing algorithms to identify the sentiment of the market. Through this, one can track the emotion towards a particular product, stock, commodity or currency and make the right investment decision.

Equity research 
Equity Research refers to analysing a company’s financial data, perform analysis over it and identify recommendations for buying and selling of stocks. The main aim of equity research is to provide investors with financial analysis reports and recommendations on buying, holding, or selling a particular investment. Also, banks and financial investment organisations often use equity research for their investments and sales & trading clients, by providing timely, high-quality information and analysis.

Regulatory compliance Business and financial investment jobs are high-risk jobs. A lot of investment decisions are directly dependent on the government scheme and policies regarding trade. Hence, it is essential to keep track of the government sites and other official forums to extract any policy changes related to trading. Mainly, risk analysts should crawl news outlets and government sites for real-time actions about the events and decisions which are directly correlated with their business.
Approach for scraping Yahoo finance data
Yahoo finance provides a plethora of information of about stock market and investment. Our primary goal is to fetch the data by scraping Yahoo finance and store it on our own premises for later analysis. In this blog, we are going to extract data about cryptocurrencies, currencies, world-indices, active-stocks and commodities. These data points can also be scraped from the results of search engine too, but we will keep the scope to scraping Yahoo finance only in this blog.

We will be writing simple python code for scraping Yahoo finance data which will visit the website and get all this data for us. Python is used for the crawler implementation. We are using the Beautiful Soup library to do crawling for us!

Python implementation for scraping Yahoo finance data
We start by importing the required libraries for us. We have imported the pandas and Beautiful Soup library here. Pandas library will help us in arranging the collected data in the form of tables whereas the Beautiful Soup library provides us with the crawling abilities in python
'''

# Scraping Crypto Currencies
# A cryptocurrency is a digital currency using cryptographic security. Cryptocurrencies are decentralised systems based on blockchain technology, a distributed network of computers. Due to advanced protection, these currencies are harder to counterfeit.

# By now, cryptocurrencies have become a global phenomenon. With significant growth in recent years, investments in cryptocurrencies prooved beneficial to a large number of investors.

# Scraping Yahoo Finance Data using Pythoncrypto currencies from scraping yahoo finance
# In below code section, we have given the yahoo finance link for the cryptocurrencies page. There are multiple pages which contain information about the cryptocurrencies. This code iterates through all the pages and pulls out the relevant information. Pulling of any relevant information happens through HTML tags present in the source code of the website. We just need to identify those tags and put them in attributes placeholder in the code!
names=[]
prices=[]
changes=[]
percentChanges=[]
marketCaps=[]
totalVolumes=[]
circulatingSupplys=[]

for i in range(0,10):
  CryptoCurrenciesUrl = "https://in.finance.yahoo.com/cryptocurrencies?offset="+str(i)+"&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;count=50"
  r= requests.get(CryptoCurrenciesUrl)
  data=r.text
  soup=BeautifulSoup(data, "lxml")


  for listing in soup.find_all('tr', attrs={'class':'SimpleDataTableRow'}):
    for name in listing.find_all('td', attrs={'aria-label':'Name'}):
      names.append(name.text)
    for price in listing.find_all('td', attrs={'aria-label':'Price (intraday)'}):
      prices.append(price.find('span').text)
    for change in listing.find_all('td', attrs={'aria-label':'Change'}):
      changes.append(change.text)
    for percentChange in listing.find_all('td', attrs={'aria-label':'% change'}):
      percentChanges.append(percentChange.text)
    for marketCap in listing.find_all('td', attrs={'aria-label':'Market cap'}):
      marketCaps.append(marketCap.text)
    for totalVolume in listing.find_all('td', attrs={'aria-label':'Total volume all currencies (24 hrs)'}):
      totalVolumes.append(totalVolume.text)
    for circulatingSupply in listing.find_all('td', attrs={'aria-label':'Circulating supply'}):
      circulatingSupplys.append(circulatingSupply.text)

# Scraping Currencies

names=[]
prices=[]
changes=[]
percentChanges=[]
marketCaps=[]
totalVolumes=[]
circulatingSupplys=[]
 
CryptoCurrenciesUrl = "https://in.finance.yahoo.com/currencies"
r= requests.get(CryptoCurrenciesUrl)
data=r.text
soup=BeautifulSoup(data, "lxml")
 
counter = 40
for i in range(40, 404, 14):
   for listing in soup.find_all('tr', attrs={'data-reactid':i}):
      for name in listing.find_all('td', attrs={'data-reactid':i+3}):
         names.append(name.text)
      for price in listing.find_all('td', attrs={'data-reactid':i+4}):
         prices.append(price.text)
      for change in listing.find_all('td', attrs={'data-reactid':i+5}):
         changes.append(change.text)
      for percentChange in listing.find_all('td', attrs={'data-reactid':i+7}):
         percentChanges.append(percentChange.text)
df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges})
df.to_csv('test1.csv')

# Scraping World Indices
# The MSCI World is a market cap weighted stock market index of 1,649 stocks from companies throughout the world. The index and their movements give an insight into the general attitude of the investing public towards companies of all sizes and industries.

# Scraping Yahoo Finance Data using PythonWorld Indices after scraping yahoo finance
# By Scraping yahoo finance, we get access to attributes of world indices like prices, percentage changes, market volume about the different world indices.

prices=[]
names=[]
changes=[]
percentChanges=[]
marketCaps=[]
totalVolumes=[]
circulatingSupplys=[]
 
CryptoCurrenciesUrl = "https://in.finance.yahoo.com/world-indices"
r= requests.get(CryptoCurrenciesUrl)
data=r.text
soup=BeautifulSoup(data, "lxml")
 
counter = 40
for i in range(40, 404, 14):
   for row in soup.find_all('tbody'):
      for srow in row.find_all('tr'):
         for name in srow.find_all('td', attrs={'class':'data-col1'}):
            names.append(name.text)
         for price in srow.find_all('td', attrs={'class':'data-col2'}):
            prices.append(price.text)
         for change in srow.find_all('td', attrs={'class':'data-col3'}):
            changes.append(change.text)
         for percentChange in srow.find_all('td', attrs={'class':'data-col4'}):
            percentChanges.append(percentChange.text)
 
df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges})
df.to_csv('test1.csv')


# Scraping most-active stocks
# The stocks on an exchange with the highest volume over a given period are the most active. Because of significantly important new information affecting the stock reaching the market, stocks usually have a higher than average trading volume. This gives investors a strong impetus to buy or sell the stock for high profits.

# Scraping Yahoo Finance Data using PythonMost active stocks from scraping yahoo finance
# Following code helps in scraping Yahoo finance about most-active stocks!

names=[]
prices=[]
changes=[]
percentChanges=[]
marketCaps=[]
totalVolumes=[]
circulatingSupplys=[]
 
for i in range(0,11):
  CryptoCurrenciesUrl = "https://in.finance.yahoo.com/most-active?offset="+str(i)+"&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;count=100"
  r= requests.get(CryptoCurrenciesUrl)
  data=r.text
  soup=BeautifulSoup(data, "lxml")
   
  for listing in soup.find_all('tr', attrs={'class':'SimpleDataTableRow'}):
     for name in listing.find_all('td', attrs={'aria-label':'Name'}):
        names.append(name.text)
     for price in listing.find_all('td', attrs={'aria-label':'Price (intraday)'}):
        prices.append(price.find('span').text)
     for change in listing.find_all('td', attrs={'aria-label':'Change'}):
        changes.append(change.text)
     for percentChange in listing.find_all('td', attrs={'aria-label':'% change'}):
        percentChanges.append(percentChange.text)
     for marketCap in listing.find_all('td', attrs={'aria-label':'Market cap'}):
        marketCaps.append(marketCap.text)
     for totalVolume in listing.find_all('td', attrs={'aria-label':'Avg vol (3-month)'}):
        totalVolumes.append(totalVolume.text)
     for circulatingSupply in listing.find_all('td', attrs={'aria-label':'Volume'}):
        circulatingSupplys.append(circulatingSupply.text)
 
df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges, "Market Cap": marketCaps, "Average Volume": totalVolumes,"Volume":circulatingSupplys})
df.to_csv('test3.csv')


# Scraping commodities
# A commodity is an essential commodity used in trade that can be exchanged with the same type of commodity. The Commodities are most frequently used as inputs to other goods or services. Traders trade in commodity markets solely to benefit from volatile price changes. These traders never intend to supply the actual commodity or take it when the futures contract expires.

# Scraping Yahoo Finance Data using PythonMost active stocks from scraping yahoo finance
# Below helps in scraping yahoo finance for the data about different commodities like gold and silver.

prices=[]
names=[]
changes=[]
percentChanges=[]
marketCaps=[]
marketTimes=[]
totalVolumes=[]
openInterests=[]
 
CryptoCurrenciesUrl = "https://in.finance.yahoo.com/commodities"
r= requests.get(CryptoCurrenciesUrl)
data=r.text
soup=BeautifulSoup(data, "lxml")
 
counter = 40
for i in range(40, 404, 14):
   for row in soup.find_all('tbody'):
      for srow in row.find_all('tr'):
         for name in srow.find_all('td', attrs={'class':'data-col1'}):
            names.append(name.text)
         for price in srow.find_all('td', attrs={'class':'data-col2'}):
            prices.append(price.text)
         for time in srow.find_all('td', attrs={'class':'data-col3'}):
            marketTimes.append(time.text)
         for change in srow.find_all('td', attrs={'class':'data-col4'}):
            changes.append(change.text)
         for percentChange in srow.find_all('td', attrs={'class':'data-col5'}):
            percentChanges.append(percentChange.text)
         for volume in srow.find_all('td', attrs={'class':'data-col6'}):
            totalVolumes.append(volume.text)
         for openInterest in srow.find_all('td', attrs={'class':'data-col7'}):
            openInterests.append(openInterest.text)
 
df = pd.DataFrame({"Names": names, "Prices": prices, "Change": changes, "% Change": percentChanges, "Market Time": marketTimes,'Open Interest': openInterests ,"Volume": totalVolumes})
df.to_csv('test4.csv')
# Also, you can find the snapshot of commodities data collected after scraping yahoo finance below.
