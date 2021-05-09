import pandas as pd
import talib
from talib import abstract
from talib import MA_Type
import numpy as np
import yfinance as yf
import datetime

necessary_columns = ['date', 'high', 'low', 'open', 'close', 'volume']

def generate_indicators(dataset, timeperiod=5, generate_target=True, reset_index=False):
    
    # To avoid changes from original dataset
    df = pd.DataFrame(dataset).copy()
    
    # To prevent index from being Date
    if(reset_index):
        df = df.reset_index()
    
    df.columns = df.columns.str.lower()
    
    # check the given dataset has necessary_columns
    __check_columns(df)
    
    df = df[necessary_columns]
    
    # df_indicators has all columns except date, this is for talib to produces other indicators
    df_indicators = df.iloc[:, 1:]
    
    
    # Produce other indicators
    RSI = abstract.RSI(df_indicators.close, timeperiod)
    RSI = pd.DataFrame({'RSI' : RSI})
    
    MOM = abstract.MOM(df_indicators.close, timeperiod)
    MOM = pd.DataFrame({'MOM':MOM})

    KD = abstract.STOCH(df_indicators)
    KD = pd.DataFrame(KD) # KD has slowd and slowk

    MACD = abstract.MACD(df_indicators) 
    MACD = pd.DataFrame(MACD) # MACD has its own column names

    ADX = abstract.ADX(df_indicators)
    ADX = pd.DataFrame({'ADX': ADX})

    SMA = abstract.SMA(df_indicators.close)
    SMA = pd.DataFrame({'SMA': SMA})

    upper, middle, lower = talib.BBANDS(df_indicators.close, matype=MA_Type.T3)
    bb_df = pd.DataFrame({ \
        'upper_bb' : upper,
        'middel_bb' : middle,
        'lower_bb' : lower})
    
    # Combine all metrix
    frames = [df, RSI, MOM, KD, MACD, ADX, SMA, bb_df]
    combined = pd.concat(frames, axis = 1)
    
    if(generate_target):
        target_name = 'next_' + str(timeperiod) + 'day_trend'
        combined[target_name] = np.where(combined.close.shift(-timeperiod) > combined.close, 1, 0)
    
    return combined

def fetch_stock_data(stock_code, period='100y'):
    stock = yf.Ticker(stock_code)
    history = stock.history(period=period)
    if len(history) == 0:
        raise ValueError('Cannot find the stock code.')
    return history
    
def __check_columns(df):
    for c in necessary_columns:
        if c not in df.columns:
            raise ValueError('The stock dataset is missing ' + str(c) + ' column')

def __date_to_timestamp(date_str, date_format):
    d = datetime.datetime.strptime(date_str, date_format)
    return d.timestamp()