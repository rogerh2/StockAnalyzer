import pandas as pd
from constants import PENNY_STOCKS
from constants import ALL_TICKERS

def tickers_per_word(TICKER_DICT, ticker_category):

    key_words = {}
    for ticker in TICKER_DICT.keys():
        current_key_words = TICKER_DICT[ticker]['key_terms']
        for word in current_key_words:
            if word in key_words.keys():
                key_words[word].append(ticker)
            else:
                key_words[word] = [ticker]

    for word in key_words:
        key_words[word].insert(0, str(len(key_words[word])))

    print('There are ' + str(len(key_words) + 2*len(TICKER_DICT)) + ' calls to the NewsAPI for this dictionary' + '\nand ' + str(len(TICKER_DICT)) + ' tickers')

    key_words_frame = pd.DataFrame.from_dict(key_words, orient='index').sort_values(0)
    key_words_frame.to_csv(r'/Users/rjh2nd/PycharmProjects/StockAnalyzer/Miscellaneous//' + ticker_category + '_key_word_usage.csv')

    return key_words_frame


if __name__ == "__main__":
    tickers_per_word(PENNY_STOCKS, 'penny_stocks')
    tickers_per_word(ALL_TICKERS, 'all_stocks')