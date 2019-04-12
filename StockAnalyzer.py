import quandl
import numpy as np
import pandas as pd
import datetime
from newsapi import NewsApiClient
from pytrends.request import TrendReq
from transform_functions import functions
from util import convert_utc_str_to_est_str

PYTREND = TrendReq(tz=300)
NEWSAPI = NewsApiClient(api_key='API_KEY')

class News:
    # TODO give the News class the ability to store info by the day

    def __init__(self, term):
        self.articles = NEWSAPI.get_top_headlines(q=term)
        utc_fmt = '%Y-%m-%dT%H:%M:%SZ'
        est_fmt = "%Y-%m-%d"
        days_list = [convert_utc_str_to_est_str(article["publishedAt"], utc_fmt, est_fmt) for article in self.articles]
        self.days = {}

    def num_articles(self):
        num_articles = len(self.articles)

        return num_articles






class KeyTerm:

    funcs = functions

    def __init__(self, term, date_range):
        # all dates formatted as YYYY-MM-DD
        self.term = term
        self.date_range = date_range
        self.data = {'GoogleTrend': PYTREND.build_payload(term, timeframe=date_range), 'News':News(term)}

    def fit_to_data(self, external_data, transorm_fun=None, data_name='GoogleTrend'):
        # external_data: external data to correlate to key term data
        # transform_fun: a function to transform the current data before fitting
        # data_name: the key for the data to fit to from the self.data dictionary

        if transorm_fun is None:
            transorm_fun = self.funcs['unity']
        internal_data = transorm_fun(self.data[data_name])
        coeff = np.polyfit(internal_data, external_data, 1)
        return coeff