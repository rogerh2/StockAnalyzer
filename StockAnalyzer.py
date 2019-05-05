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
from time import sleep
from matplotlib import pyplot as plt

PYTREND = TrendReq(tz=300)
news_api_key = input('What is the News API key?:')
alpha_vantage_api_key = input('What is the Alpha Vantage API key?:')
NEWSAPI = NewsApiClient(api_key=news_api_key)
ALPHA_TS = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
ALL_TICKERS = {
    'NAO': {'key_terms':  # noticesed trends with news polarity
                ['Marshall Islands', 'supply vessels', 'crew boats', 'anchor handling vessels'],
            'name': 'Nordic American Offshore'},
    'ROSE': {'key_terms':  # Notice strong trends with drilling googletrends and weak trends with news polarity
                 ['Delaware Basin', 'drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling'],
             'name': 'Rosehill Resources'},
    'RHE': {'key_terms':
                ['AdCare Health Systems', 'healthcare', 'senior living', 'healthcare real estate', 'real estate', 'dialysis', 'Northwest Property Holdings', 'CP Nursing', 'ADK Georgia', 'Attalla Nursing'],
            'name': 'Regional Health'},
    'MAN': {'key_terms':
                ['staffing company', 'contractor', 'proffesional services', 'business services', 'administrative services'],
            'name': 'ManpowerGroup'},
    'AMD':{'key_terms':
                ['semiconductor', 'computer', 'Apple', 'Intel', 'Microprocessor', 'NVIDIA'],
            'name': 'Advanced Micro Devices'},
    'ARA':{'key_terms':
                ['dialysis', 'renal', 'nephrologist', 'kidney disease', 'kidney failure'],
            'name': 'American Renal Associates Holdings'},
    'ADNT':{'key_terms':
                ['car', 'automotive', 'dealerships', 'used car', 'automotive seating', 'automotive supplier'],
            'name': 'Adient PLC'},
    'ASPN': {'key_terms':
                ['aerogel', 'insulation', 'energy', 'pyrogel', 'cryogel'],
             'name': 'Aspen Aerogels'},
    'TLSA': {'key_terms':
                ['Pharma', 'cancer', 'immune disease', 'inflammation', 'therapeutics'],
             'name': 'Tiziana Life Sciences'},
    'MRNA': {'key_terms':
                ['Pharma', 'cancer', 'immune disease', 'inflammation', 'mRNA', 'gene therapy', 'regenerative medicine', 'immuno-oncology', 'rare diseases'],
             'name': 'Moderna'},
    'IMTE': {'key_terms':
                ['telecommunications', 'cyber security', 'big data', 'IT services', 'media services'],
             'name': 'Integrated Media Technology'},
    'ENVA': {'key_terms':
                ['financial services', 'big data', 'loans', 'financing'],
         'name': 'Enova'},
    'FET': {'key_terms':
                ['drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling'],
         'name': 'Forum Energy Technologies'},
    'VSLR': {'key_terms':
                    ['Tesla', 'solar', 'clean energy', 'drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling'],
             'name': 'Vivint Solar'},
    'ABG': {'key_terms':
                 ['automotive', 'auto dealerships', 'cars', 'Tesla', 'used cars', 'new cars'],
             'name': 'Asbury Automotive Group'},
    'LAD': {'key_terms':
                 ['automotive', 'auto dealerships', 'cars', 'Tesla', 'used cars', 'new cars'],
             'name': 'Lithia Motors'},
    'OOMA': {'key_terms':
                 ['telecommunications', 'internet', 'cyber security', 'telephone', 'telephone service'],
             'name': 'Ooma'},
    'MX': {'key_terms':
                     ['semiconductor', 'computer', 'Apple', 'Intel', 'Microprocessor', 'NVIDIA'],
                 'name': 'MagnaChip Semiconductor'},
    'EXR': {'key_terms':
               ['self-storage', 'Marie Kondo', 'relocating', 'moving', 'long term storage'],
           'name': 'Extra Space Storage'},
    'JOE': {'key_terms':
               ['Florida real estate', 'Florida land prices', 'Florida', 'residential real estate', 'commercial real estate', 'rural land', 'Florida forests'],
           'name': 'St. Joe Co.'},
    'PRPO': {'key_terms':
                ['hospitals', 'AI and medicine', 'deep learning', 'meadical misdiagnosis', 'cancer'],
            'name': 'Precipio'}
}

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

        self.days.sort()

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
        if (len(articles) > 0):
            for article in articles:
                if not article['content'] is None:
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

    def save_news_articles(self):
        data_for_df = {'title':[], 'publishedAt':[],'url':[], 'description':[]}
        for date in self.articles.keys():
            for article in self.articles[date]:
                for key in data_for_df.keys():
                    data_for_df[key].append(article[key])

        end_date = self.days[-1]
        pd.DataFrame(data_for_df).to_csv('/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock News Articles/' + self.term + '_from ' + end_date + '.csv')

class KeyTerm:

    funcs = functions

    def __init__(self, term, date_range):
        # all dates formatted as YYYY-MM-DD
        self.term = term
        self.date_range = date_range
        PYTREND.build_payload([term], timeframe=date_range)
        self.data = {'GoogleTrend': PYTREND.interest_over_time()} # News is another entry, but should be added for every instance at once to save on API calls
        sleep(0.3)

    def convert_googletrend_index_to_str(self):
        if len(self.data['GoogleTrend'].index.values) > 0:
            current_inds = self.data['GoogleTrend'].index.values
            new_inds = [str(t)[:10] for t in current_inds]
            self.data['GoogleTrend'].index = new_inds
            is_partial = self.data['GoogleTrend']['isPartial']
            is_full = [not x for x in is_partial]
            self.data['GoogleTrend'] = self.data['GoogleTrend'][is_full].drop('isPartial', axis=1)
            return True
        return False

    def choose_df_with_latest_date(self):
        latest_ts = 0
        latest_key = 'GoogleTrend'

        for key in self.data.keys():
            ts = dt.strptime(self.data[key].index[-1], "%Y-%m-%d").timestamp()
            if ts > latest_ts:
                latest_key = key

        return latest_key

    def join_data_into_dataframes(self):
        trend_data_exists = self.convert_googletrend_index_to_str()
        start_key = None # Just here to make pycharm not highlight start_key in yellow
        if trend_data_exists:
            start_key = 'GoogleTrend'
        else:
            keys = self.data.keys()
            if len(keys) == 1:
                return False
            for key in self.data.keys():
                if key == 'GoogleTrend':
                    continue
                else:
                    start_key = key
                    break

        self.df = self.data[start_key]
        for key in self.data.keys():
            if key == start_key:
                continue
            self.df = self.df.merge(self.data[key], how='outer', left_index=True, right_index=True)

        return True

class CorporateID(KeyTerm):

    def __init__(self, name, date_range):
        super().__init__(name, date_range)
        self.news = News(name)
        self.news.create_data_frame()
        self.data['News'] = self.news.df

class Ticker(CorporateID):

    def __init__(self, ticker):
        stock_data, _ = ALPHA_TS.get_daily_adjusted(symbol=ticker)
        sleep(0.3)
        date_range = stock_data.index.values[0] + ' ' + stock_data.index.values[-1]
        super().__init__(ticker, date_range)
        self.ticker = ticker
        self.data['Price'] = stock_data

class Corporation(Ticker):

    def __init__(self, name, ticker, key_words_list, key_words_news_dfs):

        super(Corporation, self).__init__(ticker)
        self.name = name
        name_obj = CorporateID(name, self.date_range)
        name_obj.join_data_into_dataframes()
        self.data[name] = name_obj.df

        if not key_words_news_dfs is None:
            for word, news in zip(key_words_list, key_words_news_dfs):
                current_term = KeyTerm(word, self.date_range)
                current_term.data['News'] = news
                term_has_data = current_term.join_data_into_dataframes()
                if term_has_data:
                    self.data[word] = current_term.df
        else:
            for word in key_words_list:
                current_term = KeyTerm(word, self.date_range)
                term_has_data = current_term.join_data_into_dataframes()
                if term_has_data:
                    self.data[word] = current_term.df

    def save_data(self):
        self.join_data_into_dataframes()
        self.df.to_csv('/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data/' + self.name + '_from ' + self.df.index.values[-1] + '.csv')

def create_and_save_data():
    tickers_list = ['MX', 'EXR', 'JOE', 'PRPO']

    news_objects = {}

    for key in tickers_list:
        current_data = ALL_TICKERS[key]
        current_list = current_data['key_terms']
        current_data['news'] = []
        print('Getting data for ' + key)
        for term in current_list:
            print('Loading data for key word: ' + term)
            if not term in news_objects.keys():
                # This ensures the News API is only called once per turm
                news_objects[term] = News(term)
                sleep(0.3)
            if len(news_objects[term].days) > 0:
                news_objects[term].create_data_frame()
                news_objects[term].save_news_articles()
                current_data['news'].append(news_objects[term].df)

        if len(current_data['news']) > 0:
            current_corp = Corporation(current_data['name'], key, current_list, current_data['news'])
        else:
            current_corp = Corporation(current_data['name'], key, current_list, None)
        current_corp.save_data()

# TODO make jupyter notebook for viewing data

if __name__ == "__main__":
    create_and_save_data()