from alpha_vantage.timeseries import TimeSeries
import quandl
import numpy as np
import pandas as pd
import datetime
from newsapi import NewsApiClient
from pytrends.request import TrendReq
from transform_functions import functions
from util import convert_utc_str_to_est_str
from textblob import TextBlob as txb

PYTREND = TrendReq(tz=300)
news_api_key = input('What is the News API key?:')
alpha_vantage_api_key = input('What is the Alpha Vantage API key?:')
NEWSAPI = NewsApiClient(api_key=news_api_key)
ALPHA_TS = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')

class News:

    def __init__(self, term):
        articles = NEWSAPI.get_everything(q=term, language='en')
        self.articles = {}
        utc_fmt = '%Y-%m-%dT%H:%M:%S'
        est_fmt = "%Y-%m-%d"
        days_list = [convert_utc_str_to_est_str(article["publishedAt"], utc_fmt, est_fmt) for article in
                     articles['articles']]

        for day, article in zip(days_list, articles['articles']):
            if day in self.articles.keys():
                self.articles[day].append(article)
            else:
                self.articles[day] = [article]

    def __getitem__(self, day):
        if day in self.articles.keys():
            articles = self.articles[day]
        else:
            articles = []

        return articles

    def get_num_articles(self, day):
        num_articles = len(self[day])
        return num_articles

    def get_article_sentiment(self, day):
        polarity = 0
        subjectivity = 0
        articles = self[day]
        if len(articles) > 0:
            for article in articles:
                sentiment = txb(article['content']).sentiment
                polarity += sentiment.polarity
                subjectivity += sentiment.subjectivity

            mean_polarity = polarity / len(articles)
            mean_subjectivity = subjectivity / len(articles)

        else:
            mean_polarity = 0
            mean_subjectivity = 0

        return mean_polarity, mean_subjectivity

class KeyTerm:

    funcs = functions

    def __init__(self, term, date_range):
        # all dates formatted as YYYY-MM-DD
        self.term = term
        self.date_range = date_range
        PYTREND.build_payload([term], timeframe=date_range)
        self.data = {'GoogleTrend': PYTREND.interest_over_time()} # News is another entry, but should be added for every instance at once to save on API calls

    def fit_to_data(self, external_data, transorm_fun=None, data_name='GoogleTrend'):
        # external_data: external data to correlate to key term data
        # transform_fun: a function to transform the current data before fitting
        # data_name: the key for the data to fit to from the self.data dictionary
        # TODO, because data types are datarames this function does not work
        # TODO, Will probably move this functionality to a different class when possible (Stats)
        if transorm_fun is None:
            transorm_fun = self.funcs['unity']
        internal_data = transorm_fun(self.data[data_name])
        coeff = np.polyfit(internal_data, external_data, 1)
        return coeff

    # TODO add method to join all data into one dataframe for easier plotting

class Ticker(KeyTerm):

    def __init__(self, ticker):
        stock_data, _ = ALPHA_TS.get_daily_adjusted(symbol=ticker)
        date_range = stock_data.index.values[0] + ' ' + stock_data.index.values[-1]
        super().__init__(ticker, date_range)
        self.data['Price'] = stock_data

# TODO make base class for financial data: Finance (if relevant data is available)
# TODO make class for industries: Industry(KeyTerm, Finance) (if relevant data is available)
# TODO make class for products: Products(KeyTerm, Finance) (if relevant data is available)
# TODO make class for companies (that handles the key terms and the financial data): Corporation(KeyTerm, Finance)
# TODO make class to find statistical relationship between key term data and financial data for a given company: Stats

if __name__ == "__main__":
    ticker = Ticker('AAPL')
    news = News('Avengers')