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
from util import eliminate_nans_equally
from util import num2str

# --definition of global variables--
PYTREND = TrendReq(tz=300)
#TODO put api keys in a file that is automatically read when the script is executed
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
           'name': 'St. Joe Co'},
    'PRPO': {'key_terms':
                ['hospitals', 'AI and medicine', 'deep learning', 'meadical misdiagnosis', 'cancer'],
            'name': 'Precipio'}
}

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

#--definition of classes--
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
        self.df.to_csv('/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data/' + self.df.index.values[-1] + '_' + self.ticker + '_' + self.name + '.csv')

class Stats:

    def __init__(self, data_path):
        ticker = pd.read_csv(data_path, index_col=0)
        daily_change_col = ticker['4. close'] - ticker['1. open']
        daily_change_col.name = 'daily_change'
        self.ticker = ticker.join(daily_change_col)


    def create_time_dependent_arr(self, header_to_convert, t_func=inv_t, start_val=20):
        arr_to_convert = self.ticker[header_to_convert].values / np.max(self.ticker[header_to_convert].fillna(0).values)

        new_arr = np.array([])

        for i in range(start_val, len(arr_to_convert)):
            current_term = 0#arr_to_convert[i]
            for j in range(0, start_val - 1):
                t = start_val - j
                current_term += arr_to_convert[i-t] * t_func(t)

            new_arr = np.append(new_arr, current_term)

        return new_arr

    def fit_time_dependent_arr_to_arr(self, t_dependent_arr_header, arr_header, t_func=inv_t, start_val=20):
        arr = self.ticker[arr_header].values[start_val::]
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
        ticker = self.ticker
        corr_header = '1. open'
        corr_data = ticker.corr()[corr_header]
        indicators = {}

        for data_name in corr_data.index.values:
            if ('.' in data_name) or ('_' in data_name):
                continue

            current_corr = corr_data.loc[data_name]
            if (np.abs(current_corr) < 0.4):
                continue

            current_t_dependent_corr = self.find_correlation_coeff_for_time_depedent_arr_vs_arr(data_name, corr_header)
            if (current_t_dependent_corr > np.abs(current_corr)):
                # TODO generally clean this up
                news_polarity = np.sum(self.ticker[data_name + '_polarity'].dropna().values * self.ticker[data_name + '_num_articles'].dropna().values) / np.sum(self.ticker[data_name + '_num_articles'].dropna().values)
                polarity_corr = corr_data.loc[data_name + '_polarity']
                news_subjectivity = np.sum(self.ticker[data_name + '_subjectivity'].dropna().values * self.ticker[data_name + '_num_articles'].dropna().values) / np.sum(self.ticker[data_name + '_num_articles'].dropna().values)
                subjectivity_corr = corr_data.loc[data_name + '_subjectivity']
                prior_change = 100 * self.ticker.daily_change.dropna().values[-1] / self.ticker[corr_header].dropna().values[-1]
                prior_change_corr = self.ticker.daily_change.autocorr()

                indicators[data_name] = {'Polarity':(news_polarity, polarity_corr), 'Subjectivity':(news_subjectivity, subjectivity_corr), 'Google Trend Correlation':current_t_dependent_corr}
                indicators['Previous Movement'] = (prior_change, prior_change_corr)

        return indicators

    def analyze_correlations(self, corr_header):
        ticker = self.ticker
        corr_data = ticker.corr()[corr_header]

        for data_name in corr_data.index.values:
            if (np.abs(corr_data.loc[data_name]) > 0.4) and (np.abs(corr_data.loc[data_name]) < 1):
                ticker.plot(x=data_name, y=corr_header, style=['rx'])
                plt.title(data_name + 'vs ' + corr_header + ' ' + num2str(corr_data.loc[data_name], 2))

        plt.show()

    def plot_autocorrelation(self, header):
        data = self.ticker[header]
        plt.figure()
        plt.plot(data.values[0:-1], data.values[1::], 'rx')
        plt.title('Daily Change Autocorellation: ' + num2str(data.autocorr(), 3))
        plt.show()

#--definition of functions--

def create_and_save_data(tickers_list):

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

def print_indicators(ticker, indicators):
    print('\n' + ticker)
    print('Previous Movement' + ' - value: ' + num2str(indicators['Previous Movement'][0], 3) + '% | correlation coefficient: ' + num2str(
        indicators['Previous Movement'][1], 3))
    for key in indicators.keys():
        if ('Previous' in key):
            continue
        print('-' + key + '-')
        current_ind = indicators[key]
        print('Google Trend Correlation: ' + num2str(current_ind['Google Trend Correlation'], 3))
        for ind in current_ind.keys():
            if ('Google' in ind):
                continue
            print(ind + ' - value: ' + num2str(current_ind[ind][0], 3) + ' | correlation coefficient: '+ num2str(current_ind[ind][1], 3))

    print('--------------------------------------------------------------------')

def print_indicators_for_many_symbols(tickers_list, date, folder_path):

    for ticker in tickers_list:
        fname = folder_path + '/' +date + '_' + ticker + '_' + ALL_TICKERS[ticker]['name'] + '.csv'
        current_stats = Stats(fname)
        current_indicators = current_stats.check_indicators()
        if len(current_indicators.keys()) == 0:
            continue
        else:
            print_indicators(ticker, current_indicators)

if __name__ == "__main__":
    # create_and_save_data(['JOE', 'PRPO'])

    # stat = Stats('/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data/2019-04-29_ASPN_Aspen Aerogels.csv')
    # _ = stat.analyze_correlations('1. open')
    # stat.plot_time_dependent_arr_vs_arr('energy', '1. open', t_func=inv_t)
    # print(np.mean(stat.ticker['energy_polarity'].dropna().values * stat.ticker['energy_num_articles'].dropna().values))
    # ind = stat.check_indicators()
    print_indicators_for_many_symbols(['ROSE', 'RHE', 'MAN', 'AMD', 'ARA', 'ASPN', 'TLSA', 'MRNA', 'IMTE', 'ENVA', 'FET', 'VSLR', 'OOMA', 'MX', 'EXR'], '2019-05-05', '/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data')