from alpha_vantage.timeseries import TimeSeries
import quandl
import numpy as np
import pandas as pd
from datetime import datetime as dt
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
        self.days = []

        for day, article in zip(days_list, articles['articles']):
            if day in self.articles.keys():
                self.articles[day].append(article)
            else:
                self.days.append(day)
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

    def create_data_frame(self):
        df_data = {'num_articles':np.array([]), 'polarity':np.array([]), 'subjectivity':np.array([])}
        for day in self.days:
            df_data['num_articles'] = np.append(df_data['num_articles'], self.get_num_articles(day))
            mean_polarity, mean_subjectivity = self.get_article_sentiment(day)
            df_data['polarity'] = np.append(df_data['polarity'], mean_polarity)
            df_data['subjectivity'] = np.append(df_data['subjectivity'], mean_subjectivity)

        self.df = pd.DataFrame(data=df_data, index=self.days)

class KeyTerm:

    funcs = functions

    def __init__(self, term, date_range):
        # all dates formatted as YYYY-MM-DD
        self.term = term
        self.date_range = date_range
        PYTREND.build_payload([term], timeframe=date_range)
        self.data = {'GoogleTrend': PYTREND.interest_over_time()} # News is another entry, but should be added for every instance at once to save on API calls

    def convert_googletrend_index_to_str(self):
        current_inds = self.data['GoogleTrend'].index.values
        new_inds = [str(t)[:10] for t in current_inds]
        self.data['GoogleTrend'].index = new_inds

    def join_data_into_dataframe(self):
        self.convert_googletrend_index_to_str()
        self.df = self.data['GoogleTrend']
        for key in self.data.keys():
            if key == 'GoogleTrend':
                continue
            self.df = self.df.join(self.data[key])

class Ticker(KeyTerm):

    def __init__(self, ticker):
        stock_data, _ = ALPHA_TS.get_daily_adjusted(symbol=ticker)
        date_range = stock_data.index.values[0] + ' ' + stock_data.index.values[-1]
        super().__init__(ticker, date_range)
        self.news = News(ticker)
        self.news.create_data_frame()
        self.data['Price'] = stock_data
        self.data['News'] = self.news.df

# TODO make base class for financial data: Finance (if relevant data is available)
# TODO make class for industries: Industry(KeyTerm, Finance) (if relevant data is available)
# TODO make class for products: Products(KeyTerm, Finance) (if relevant data is available)
# TODO make class for companies (that handles the key terms and the financial data): Corporation(KeyTerm, Finance)
# TODO make class to find statistical relationship between key term data and financial data for a given company: Stats

if __name__ == "__main__":
    ticker = Ticker('AAPL')
    ticker.join_data_into_dataframe()
    news = News('Avengers')