# You will likely in time want to do some sort of force_data_update parameter to this function, since, right now, it will not re-pull data it already sees hit has. Since we're pulling daily data, you'd want to have this re-pulling at least the latest data. That said, if that's the case, you might be better off with using a database instead with a table per company, and then just pulling the most recent values from the Yahoo database. We'll keep things simple for now though!
# Full code up to this point:

import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'morningstar', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo()


#fix
# https://github.com/pydata/pandas-datareader/issues/487
# import fix_yahoo_finance as yf
# yf.pdr_override()

# https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/?completed=/sp500-company-list-python-programming-for-finance/