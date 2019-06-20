import os
import re
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime as dt
from datetime import timedelta
from newsapi import NewsApiClient
from pytrends.request import TrendReq
from transform_functions import functions
from textblob import TextBlob as txb
from time import sleep
from matplotlib import pyplot as plt
from util import convert_utc_str_to_est_str
from util import eliminate_nans_equally
from util import num2str
from util import plot_informative_lines
from util import ClassifierNN
from util import progress_printer
from constants import ALL_TICKERS
from constants import PENNY_STOCKS
from constants import STOCK_DATA_PATH
from constants import NEXT_TICKERS
from constants import FMT
from constants import NN_TRAINING_DATA_PATH
from constants import MODEL_PATH
from constants import DATA_PATH

# --definition of global variables--
PYTREND = TrendReq(tz=300)
INPUT_SIZE=17
#TODO put api keys in a file that is automatically read when the script is executed
api_keys_file_path = '/Users/rjh2nd/PycharmProjects/StockAnalyzer/APIKeys.txt'
api_keys_file_exists = os.path.isfile(api_keys_file_path)
if api_keys_file_exists:
    with open(api_keys_file_path, 'r') as api_keys:
        all_keys = api_keys.readlines()
        news_api_key = re.search('(?<=News API Key: ).*', all_keys[0])[0]
        alpha_vantage_api_key = re.search('(?<=Alpha-Vantage API Key: ).*', all_keys[2])[0]
else:
    news_api_key = input('What is the News API key?:')
    alpha_vantage_api_key = input('What is the Alpha Vantage API key?:')
NEWSAPI = NewsApiClient(api_key=news_api_key)
ALPHA_TS = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')

# --functions used for analysis--
def inv_t(t):
    ans = 1 / t
    return ans

def inv_sq_t(t):
    ans = -inv_t(t**2)
    return ans

def exp_decay(t):
    ans = np.exp(-t)
    return ans

def unity(t):
    return t

def create_between_day_change_col(df):
    opens = df['1. open'].values
    closes = df['4. close'].values
    between_day_change = np.array([np.nan])
    for i in range(1, len(opens)):
        open = opens[i]
        if np.isnan(open):
            between_day_change = np.append(between_day_change, np.nan)
            continue

        close = np.nan
        j = 1
        while np.isnan(close) and (j <= i):
            close = closes[i-j]
            j += 1

        between_day_change = np.append(between_day_change, open - close)

    return between_day_change



#--definition of classes--

class News:

    def __init__(self, term):
        self.term = term
        articles = NEWSAPI.get_everything(q=term, language='en')
        self.articles = {}
        utc_fmt = '%Y-%m-%dT%H:%M:%S'
        est_fmt = FMT
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
        self.df.to_csv(STOCK_DATA_PATH + self.df.index.values[-1] + '_' + self.ticker + '_' + self.name + '.csv')

class Stats:

    def __init__(self, data_path):
        stock_data = pd.read_csv(data_path, index_col=0)
        daily_change_col = stock_data['4. close'] - stock_data['1. open']
        daily_change_col.name = 'daily_change'
        between_day_change_col =  pd.Series(data=create_between_day_change_col(stock_data), index=daily_change_col.index)
        between_day_change_col.name = 'between_day_change'
        stock_data = stock_data.join(daily_change_col)
        self.stock_data = stock_data.join(between_day_change_col)

    def create_time_dependent_arr(self, header_to_convert, t_func=inv_t, start_val=20):
        arr_to_convert = self.stock_data[header_to_convert].values / np.max(self.stock_data[header_to_convert].fillna(0).values)

        new_arr = np.array([])
        current_term = 0

        for i in range(start_val, len(arr_to_convert)):
            if np.isnan(arr_to_convert[i]):
                new_arr = np.append(new_arr, current_term)
                continue
            current_term = 0#arr_to_convert[i]
            for j in range(0, start_val - 1):
                t = start_val - j
                if not np.isnan(arr_to_convert[i-t]):
                    current_add = arr_to_convert[i-t] * t_func(t)
                    current_term += current_add

            new_arr = np.append(new_arr, current_term)

        return new_arr

    def fit_time_dependent_arr_to_arr(self, t_dependent_arr_header, arr_header, t_func=inv_t, start_val=20):
        arr = self.stock_data[arr_header].values[start_val::]
        t_dependent_arr = self.create_time_dependent_arr(t_dependent_arr_header, t_func=t_func, start_val=start_val)
        no_nan_arrs = eliminate_nans_equally([arr, t_dependent_arr])
        daily_open_no_nan = no_nan_arrs[0]
        header_arr_no_nan = no_nan_arrs[1]

        coeff = np.polyfit(header_arr_no_nan, daily_open_no_nan, 1)

        return coeff, t_dependent_arr, arr

    def plot_time_dependent_arr_vs_arr(self, t_dependent_arr_header, arr_header, t_func=inv_t, start_val=20):
        coeff, t_dependent_arr, arr = self.fit_time_dependent_arr_to_arr(t_dependent_arr_header, arr_header, t_func=t_func, start_val=start_val)
        plt.plot(arr - coeff[0] * t_dependent_arr, '--bo')
        plt.title('Zeroed')
        plt.figure()
        plt.plot(coeff[0] * t_dependent_arr, arr, 'rx')
        plt.title('Fit')
        plt.figure()
        plt.plot(arr)
        plt.title(arr_header)
        plt.figure()
        plt.plot(t_dependent_arr)
        plt.title(t_dependent_arr_header)

        plt.show()

    def find_correlation_coeff_for_time_depedent_arr_vs_arr(self, t_dependent_arr_header, arr_header, t_func=inv_t, start_val=20):
        coeff, t_dependent_arr, arr = self.fit_time_dependent_arr_to_arr(t_dependent_arr_header, arr_header, t_func=t_func, start_val=start_val)
        df = pd.DataFrame({'timde dependent array': coeff[0] * t_dependent_arr, 'array':arr})
        corr = df.corr().loc['array'].values[0]

        return corr

    def check_indicators(self):
        ticker = self.stock_data
        corr_header = '1. open'
        corr_data = ticker.corr()[corr_header]
        indicators = {}

        for data_name in corr_data.index.values:

            if ('_' in data_name) or (data_name == '4. close'):
                continue

            if ('.' in data_name):
                current_change = self.stock_data[data_name].dropna().values[-1] - self.stock_data[data_name].dropna().values[-2]
                if 'volume' in data_name:
                    coeff = 1
                else:
                    coeff = 100
                data_percent = coeff * current_change / self.stock_data[data_name].dropna().values[-2]
                indicators['Previous ' + data_name[3::].capitalize()] = (data_percent, corr_data.loc[data_name])
                continue

            current_corr = corr_data.loc[data_name]
            if (np.abs(current_corr) < 0.25):
                continue

            if not (data_name + '_polarity') in self.stock_data.keys():
                continue

            current_t_dependent_corr = self.find_correlation_coeff_for_time_depedent_arr_vs_arr(data_name, corr_header)
            # if (np.abs(current_t_dependent_corr) < np.abs(current_corr)):
            #     current_t_dependent_corr = current_corr
                # TODO generally clean this up
            news_polarity = np.sum(self.stock_data[data_name + '_polarity'].dropna().values * self.stock_data[data_name + '_num_articles'].dropna().values) / np.sum(self.stock_data[data_name + '_num_articles'].dropna().values)
            polarity_corr = corr_data.loc[data_name + '_polarity']
            news_subjectivity = np.sum(self.stock_data[data_name + '_subjectivity'].dropna().values * self.stock_data[data_name + '_num_articles'].dropna().values) / np.sum(self.stock_data[data_name + '_num_articles'].dropna().values)
            subjectivity_corr = corr_data.loc[data_name + '_subjectivity']
            indicators[data_name] = {'Polarity':(news_polarity, polarity_corr), 'Subjectivity':(news_subjectivity, subjectivity_corr), 'Google Trend Correlation':current_t_dependent_corr}
        prior_change = 100 * self.stock_data.daily_change.dropna().values[-1] / self.stock_data[corr_header].dropna().values[-1]
        prior_change_corr = self.stock_data.daily_change.autocorr()
        indicators['Previous Movement'] = (prior_change, prior_change_corr)

        return indicators

    def analyze_correlations(self, corr_header):
        ticker = self.stock_data
        corr_data = ticker.corr()[corr_header]

        for data_name in corr_data.index.values:
            if (np.abs(corr_data.loc[data_name]) > 0.4) and (np.abs(corr_data.loc[data_name]) < 1):
                ticker.plot(x=data_name, y=corr_header, style=['rx'])
                plt.title(data_name + 'vs ' + corr_header + ' ' + num2str(corr_data.loc[data_name], 2))

        plt.show()

    def plot_autocorrelation(self, header):
        data = self.stock_data[header]
        plt.figure()
        plt.plot(data.values[0:-1], data.values[1::], 'rx')
        plt.title('Daily Change Autocorellation: ' + num2str(data.autocorr(), 3))
        plt.show()

class MultiSymbolStats:

    def __init__(self, tickers_list, date, folder_path=STOCK_DATA_PATH):
        self.stats = {}
        for ticker in tickers_list:
            fname = folder_path + date + '_' + ticker + '_' + ALL_TICKERS[ticker]['name'] + '.csv'
            file_does_not_exist = not os.path.isfile(fname)
            if file_does_not_exist:
                continue
            self.stats[ticker] = Stats(fname)

    def get_next_daily_change_percentage(self, ticker, day):
        #TODO remove dual function
        corp_name = ALL_TICKERS[ticker]['name']
        current_fname_day = dt.strptime(day, FMT)
        current_day = current_fname_day
        give_up_counter = 0
        # The outer loop finds files that should have the next day stored
        while give_up_counter < 100:
            give_up_counter += 1
            current_fname_day = current_fname_day + timedelta(days=1)
            current_fname_day_str = current_fname_day.strftime(FMT)
            current_fname = STOCK_DATA_PATH + '_'.join([current_fname_day_str, ticker, corp_name]) + '.csv'
            file_exists = os.path.isfile(current_fname)
            if not file_exists:
                continue
            current_day_df = Stats(current_fname).stock_data

            # The inner loop finds files the next day with data in the chosen file
            still_searching = True
            days = current_day_df.index.values
            while still_searching:
                current_day = current_day + timedelta(days=1)
                current_day_str = current_day.strftime(FMT)
                if not current_day_str in days:
                    break
                change = 100 * current_day_df.daily_change.loc[current_day_str] / current_day_df['1. open'].loc[
                    current_day_str]
                if not np.isnan(change):
                    return change, current_day_str

        return None, None

    def create_indicator_rows_for_symbol(self, ticker):
        indicators = self.stats[ticker].check_indicators()
        if len(indicators) <= 8:
            return None
        data_headers = ['Polarity', 'Polarity Correlation', 'Subjectivity', 'Subjectivity Correlation', 'Google Trend Correlation']
        data = {}
        initial_value = np.array([])
        key_words = []
        for header in data_headers:
            data[header] = initial_value

        for key in indicators.keys():
            if ('Split' in key) or ('Dividend' in key):
                continue
            if ('Previous' in key):
                if key in data.keys():
                    data[key] = np.append(data[key], indicators[key][0])
                    data[key + ' Correlation'] = np.append(data[key], indicators[key][1])
                else:
                    data[key] = np.array([indicators[key][0]])
                    data[key + ' Correlation'] = np.array([indicators[key][1]])
                continue

            key_words.append(ticker + '-' + key)
            current_ind = indicators[key]
            for ind in current_ind.keys():
                if not 'Correlation' in ind:
                    data[ind] = np.append(data[ind], current_ind[ind][0])
                    data[ind + ' Correlation'] = np.append(data[ind + ' Correlation'], current_ind[ind][1])
                else:
                    data[ind] = np.append(data[ind], current_ind[ind])

        df = pd.DataFrame(data, index=key_words)
        return df

    def creat_indicator_dfs(self):
        df = None
        for ticker in self.stats.keys():
            current_df = self.create_indicator_rows_for_symbol(ticker)
            if current_df is None:
                continue
            if df is None:
                df = current_df
            else:
                df = pd.concat((df, current_df))

        return df

    def score_one_arr(self, raw_arr, corr_arr):
        corr_mask = np.abs(corr_arr) > 0.1
        score = np.sum(corr_mask * raw_arr * corr_arr / np.abs(raw_arr * corr_arr))
        return score

    def score_tickers(self):
        inds = []
        scores = {'news': [], 'recent_movement': []}
        for ticker in self.stats.keys():
            current_df = self.create_indicator_rows_for_symbol(ticker)
            if current_df is None:
                continue
            pol_score = self.score_one_arr(current_df['Polarity'].values, current_df['Polarity Correlation'].values)
            sub_score = self.score_one_arr(current_df['Subjectivity'].values,
                                           current_df['Subjectivity Correlation'].values)
            scores['news'].append(pol_score + sub_score)
            scores['recent_movement'].append(current_df['Previous Movement'].values[0])
            inds.append(ticker)
        df = pd.DataFrame(data=scores, index=inds)

        return df

    #TODO finish this
    # def compare_ticker_to_gains(self):
    #     df = self.score_tickers()
    #     for ticker in df.index.values:


    def plot_score(self, news_ax=None, mvmt_ax=None):
        df = self.score_tickers()

        if news_ax is None:
            fig, axes = plt.subplots(nrows=2, ncols=1)
            news_ax = axes[0]
            mvmt_ax = axes[1]
            news_ax.set_title('News Score')
            mvmt_ax.set_title('Most Recent Daily Change')
        df.news.plot.bar(ax=news_ax)
        plt.sca(news_ax)
        plt.ylabel('Score')
        news_ax.xaxis.set_ticklabels([])
        df.recent_movement.plot.bar(ax=mvmt_ax)
        plt.sca(mvmt_ax)
        plt.ylabel("Last Day's Movement (%)")

        return news_ax, mvmt_ax

    def print_indicators(self, ticker, indicators):
        print('\n' + ticker)
        print('Previous Movement' + ' - value: ' + num2str(indicators['Previous Movement'][0], 3) + '% | correlation coefficient: ' + num2str(
            indicators['Previous Movement'][1], 3))
        for key in indicators.keys():
            if ('Previous' in key):
                continue
            print('-' + key + '-')
            current_ind = indicators[key]
            print('Google Trend Correlation: ' + num2str(current_ind['Google Trend Correlation'], 3))
            news_score = 0
            for ind in current_ind.keys():
                if ('Google' in ind):
                    continue
                print(ind + ' - value: ' + num2str(current_ind[ind][0], 3) + ' | correlation coefficient: '+ num2str(current_ind[ind][1], 3))
                news_score += current_ind[ind][0]*current_ind[ind][1] * current_ind['Google Trend Correlation']

        print('-------------------------------------------------------------------')

    def print_indicators_for_many_symbols(self):
        # TODO add suggested ranking
        for ticker in self.stats.keys():
            stat = self.stats[ticker]
            current_indicators = stat.check_indicators()
            if len(current_indicators.keys()) == 0:
                continue
            else:
                self.print_indicators(ticker, current_indicators)

#--definition of functions--

def create_and_save_data(tickers_list):
    # TODO use multiprocessing to speedup
    news_objects = {}

    for i in range(0, len(tickers_list)):
        key = tickers_list[i]
        current_data = ALL_TICKERS[key]
        current_list = current_data['key_terms']
        current_data['news'] = []
        print('Getting data for ' + key + ' file ' + str(i + 1) + ' out of ' + str(len(tickers_list)))
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

def get_next_daily_change_percentage(ticker, day, next_header='daily_change'):
    corp_name = ALL_TICKERS[ticker]['name']
    current_fname_day = dt.strptime(day, FMT)
    current_day = current_fname_day
    give_up_counter = 0
    # The outer loop finds files that should have the next day stored
    while give_up_counter < 100:
        give_up_counter += 1
        current_fname_day = current_fname_day + timedelta(days=1)
        current_fname_day_str = current_fname_day.strftime(FMT)
        current_fname = STOCK_DATA_PATH +  '_'.join([current_fname_day_str, ticker, corp_name]) + '.csv'
        file_exists = os.path.isfile(current_fname)
        if not file_exists:
            continue
        current_day_df = Stats(current_fname).stock_data

        # The inner loop finds files the next day with data in the chosen file
        still_searching = True
        days = current_day_df.index.values
        while still_searching:
            current_day = current_day + timedelta(days=1)
            current_day_str = current_day.strftime(FMT)
            if not current_day_str in days:
                break
            change = 100 * current_day_df[next_header].loc[current_day_str] / current_day_df['1. open'].loc[current_day_str]
            if not np.isnan(change):
                return change, current_day_str

    return None, None

def create_training_output(df, day, next_header='daily_change'):
    answer_array_header = 'Next Change'
    df_data = {answer_array_header:np.array([])}
    for ind in df.index.values:
        ticker = ind.split('-')[0]
        next_percent, _ = get_next_daily_change_percentage(ticker, day, next_header=next_header)
        df_data[answer_array_header] = np.append(df_data[answer_array_header], next_percent)

    ans_df = pd.DataFrame(df_data, index=df.index)
    return ans_df

def create_training_df(days, task, next_header='daily_change', tickers=ALL_TICKERS.keys()):
    full_df = None
    for day, i in zip(days, range(0, len(days))):
        progress_printer(len(days), i, tsk=task)
        current_stats = MultiSymbolStats(tickers, day)
        current_input = current_stats.creat_indicator_dfs()
        if current_input is None:
            continue
        current_output = create_training_output(current_input, day, next_header=next_header)
        current_df = current_input.join(current_output)#.reset_index(drop=True)
        if full_df is None:
            full_df = current_df
        else:
            full_df = pd.concat((full_df, current_df))#, ignore_index=True)

    return full_df

def plot_pos_neg_groups(x_statement, y_statement, df, cutoff_percentage=0.0, ans_header='Next Change'):
    x = 1
    y = x
    for xheader, yheader in zip(x_statement.split('*'), y_statement.split('*')):
        x = x * df[xheader]
        y = y * df[yheader]

    low_mask = df[ans_header] < cutoff_percentage
    high_mask = df[ans_header] >= cutoff_percentage
    plt.plot(x[low_mask], y[low_mask], 'ro')
    plt.plot(x[high_mask], y[high_mask], 'bo')
    ax = plt.gca()
    plot_informative_lines(ax, horz_pos=0, vert_pos=0)

def check_score_vs_mvmt(score_date, mv_date):
    score_stats = MultiSymbolStats(ALL_TICKERS.keys(), score_date)
    move_stats = MultiSymbolStats(ALL_TICKERS.keys(), mv_date)
    score_df = score_stats.score_tickers()
    move_df = move_stats.score_tickers()
    new_df_data = {}

    for ticker in score_df.index.values:
        if ticker in move_df.index.values:
            score = score_df.news[ticker]
            mv = move_df.recent_movement[ticker]
            if score in new_df_data.keys():
                new_df_data[score] = np.append(new_df_data[score], mv)
            else:
                new_df_data[score] = np.array([mv])

    data = np.array([])
    inds = data
    for score in new_df_data.keys():
        data = np.append(data, np.sum(new_df_data[score] > 0))
        inds = np.append(inds, score)

    df = pd.DataFrame(data=data, index=inds)
    df.plot.bar()

def normalize_rows(df, col_list, norm_list):
    for col, norm in zip(col_list, norm_list):
        if type(norm) == str:
            df[col] = df[col].values / df[norm]
        else:
            df[col] = df[col] / norm

    return df

def save_pred_data_as_csv(df, folder_path, xlsx_name):
    # Create the Excel file
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    writer = pd.ExcelWriter(folder_path + xlsx_name + '.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    # Light red fill with dark red text.
    red_format = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})
    # Light yellow fill with dark yellow text.
    yellow_format = workbook.add_format({'bg_color': '#FFEB9C',
                                   'font_color': '#9C6500'})
    # Green fill with dark green text.
    green_format = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})

    sheet_len = str(df.shape[0] + 1)

    # Wroksheet conditions
    worksheet.conditional_format('B2:D' + sheet_len, {'type': 'data_bar'})
    worksheet.conditional_format('E2:E' + sheet_len, {'type': '3_color_scale'})
    worksheet.conditional_format('A2:A' + sheet_len,
                                 {'type': 'formula',
                                  'criteria': '=$B2>$E2',
                                  'format': red_format
                                  })
    worksheet.conditional_format('A2:A' + sheet_len,
                                 {'type': 'formula',
                                  'criteria': '=AND($E2>$B2, $C2>$D2)',
                                  'format': yellow_format
                                  })
    worksheet.conditional_format('A2:A' + sheet_len,
                                 {'type': 'formula',
                                  'criteria': '=AND($E2>$B2, $C2<$D2)',
                                  'format': green_format
                                  })
    writer.save()

if __name__ == "__main__":
    task = 'predict'
    day = '2019-06-19'

    if task == 'get_data':
        tickers = list(PENNY_STOCKS.keys())#list(ALL_TICKERS.keys())
        create_and_save_data(tickers)

    elif task == 'score_data':
        all_stats = MultiSymbolStats(ALL_TICKERS.keys(), day)
        news_ax, mvmt_ax = all_stats.plot_score()
        all_stats.print_indicators_for_many_symbols()
        plot_informative_lines(news_ax, style='k-')
        plot_informative_lines(mvmt_ax, style='k-')
        plt.show()

    elif task == 'create_training_data':
        output_header = 'daily_change'
        train_df = create_training_df(['2019-06-04', '2019-06-08', '2019-05-01', '2019-04-25', '2019-05-21', '2019-04-29', '2019-05-23', '2019-06-03', '2019-05-09', '2019-04-30', '2019-05-15', '2019-05-13', '2019-05-27', '2019-04-24', '2019-05-29', '2019-05-07', '2019-06-02', '2019-06-07', '2019-05-14', '2019-06-05', '2019-05-08', '2019-04-27', '2019-05-03', '2019-05-04', '2019-06-10', '2019-05-11', '2019-06-14', '2019-06-13', '2019-05-22', '2019-04-28', '2019-06-11', '2019-06-01', '2019-06-12', '2019-05-28', '2019-05-30'], 'creating training data', next_header=output_header, tickers=PENNY_STOCKS.keys())
        val_df = create_training_df(['2019-05-19', '2019-06-06', '2019-04-26', '2019-05-10', '2019-05-20', '2019-05-12', '2019-06-09', '2019-05-16', '2019-05-18'], 'creating validation data', next_header='between_day_change', tickers=PENNY_STOCKS.keys())
        test_df = create_training_df(['2019-05-31', '2019-05-06', '2019-05-05', '2019-05-02', '2019-04-23'], 'creating test data', next_header=output_header, tickers=PENNY_STOCKS.keys())
        train_df.to_csv(NN_TRAINING_DATA_PATH + output_header + '_training_data.csv')
        val_df.to_csv(NN_TRAINING_DATA_PATH + output_header + '_val_data.csv')
        test_df.to_csv(NN_TRAINING_DATA_PATH + output_header + '_test_data.csv')

    elif task == 'predict':
        # create dataset for predicting
        dataset = create_training_df([day], 'creating prediction data', next_header='daily_change', tickers=PENNY_STOCKS.keys())
        dataset = normalize_rows(dataset, ['Previous Movement'], [100])
        X = dataset.values[:, 0:INPUT_SIZE]
        # predict
        model_name = 'stock_daily_change_predictor_20190610.h5'
        class_nn = ClassifierNN( model_path=MODEL_PATH + model_name)
        prediction = class_nn.model.predict(X)

        # save data
        pred_df = pd.DataFrame(data=prediction, columns=['negative', 'weak positive', 'strong positive'],
                               index=dataset.index.values)
        pred_df['total positive'] = pred_df['weak positive'].values + pred_df['strong positive'].values
        save_pred_data_as_csv(pred_df,DATA_PATH + day.replace('-', '') + '/', 'Predictions_' + day.replace('-', '') + '_' + model_name[0:-3])

    elif task == 'other':
        stat = Stats('/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data/2019-04-29_ASPN_Aspen Aerogels.csv')
        _ = stat.analyze_correlations('1. open')
        stat.plot_time_dependent_arr_vs_arr('1. open', '1. open', t_func=inv_t)
        change, current_day_str = get_next_daily_change_percentage('AMD', '2019-04-30')
        check_score_vs_mvmt('2019-05-13', '2019-05-14')
        plt.show()