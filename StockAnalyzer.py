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
        self.term = term
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

    def get_data(self, key, day):
        if 'num_articles' in key:
            data = self.get_num_articles(day)
        elif 'polarity' in key:
            data, _ = self.get_article_sentiment(day)
        else:
            _, data = self.get_article_sentiment(day)

        return data



    def create_data_frame(self):
        df_data = {self.term + '_num_articles':np.array([]), self.term + '_polarity':np.array([]), self.term + '_subjectivity':np.array([])}
        for day in self.days:
            for key in df_data:
                data  = self.get_data(key, day)
                df_data[key] = np.append(df_data[key], data)

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
        is_partial = self.data['GoogleTrend']['isPartial']
        is_full = [not x for x in is_partial]
        self.data['GoogleTrend'] = self.data['GoogleTrend'][is_full].drop('isPartial', axis=1)

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
        self.ticker = ticker
        self.news = News(ticker)
        self.news.create_data_frame()
        self.data['Price'] = stock_data
        self.data['News'] = self.news.df

class Corporation(Ticker):

    def __init__(self, name, ticker, key_words_list, key_words_news_dfs):

        super(Corporation, self).__init__(ticker)
        self.name = name
        for word, news in zip(key_words_list, key_words_news_dfs):
            current_term = KeyTerm(word, self.date_range)
            current_term.data['News'] = news
            current_term.join_data_into_dataframe()
            self.data[word] = current_term.df


# TODO make base class for financial data: Finance (if relevant data is available)
# TODO make class for industries: Industry(KeyTerm, Finance) (if relevant data is available)
# TODO make class for products: Products(KeyTerm, Finance) (if relevant data is available)
# TODO make class for companies (that handles the key terms and the financial data): Corporation(KeyTerm, Finance)
# TODO make class to find statistical relationship between key term data and financial data for a given company: Stats

if __name__ == "__main__":
    iPhone_news = News('iPhone')
    iPhone_news.create_data_frame()
    Mac_news = News('Mac')
    Mac_news.create_data_frame()
    apple = Corporation('Apple', 'AAPL', ['iPhone', 'Mac'], [iPhone_news.df, Mac_news.df])
    apple.join_data_into_dataframe()
    news = News('Avengers')