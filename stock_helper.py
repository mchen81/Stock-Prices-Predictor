import pandas as pd
import talib
from talib import abstract
from talib import MA_Type
import numpy as np

necessary_columns = ['high', 'low', 'open', 'close', 'volume']

def generate_indicators(dataset, timeperiod=5, generate_target=True):
    df = pd.DataFrame(dataset).copy()
    df.columns = df.columns.str.lower()
    
    __check_columns(df)
    
    df = df[necessary_columns]
    
    RSI = abstract.RSI(df.close, timeperiod)
    RSI = pd.DataFrame({'RSI' : RSI})
    
    MOM = abstract.MOM(df.close, timeperiod)
    MOM = pd.DataFrame({'MOM':MOM})

    KD = abstract.STOCH(df)
    KD = pd.DataFrame(KD) # KD has slowd and slowk

    MACD = abstract.MACD(df) 
    MACD = pd.DataFrame(MACD) # MACD has its own column names

    ADX = abstract.ADX(df)
    ADX = pd.DataFrame({'ADX': ADX})

    SMA = abstract.SMA(df.close)
    SMA = pd.DataFrame({'SMA': SMA})

    upper, middle, lower = talib.BBANDS(df.close, matype=MA_Type.T3)
    bb_df = pd.DataFrame({ \
        'upper_bb' : upper,
        'middel_bb' : middle,
        'lower_bb' : lower})

    frames = [df,RSI, MOM, KD, MACD, ADX, SMA, bb_df]
    combined = pd.concat(frames, axis = 1)
    
    if(generate_target):
        target_name = 'passed_' + str(timeperiod) + 'day_trend'
        combined[target_name] = np.where(combined.close.shift(-timeperiod) > combined.close, 1, 0)
    
    return combined
    
def __check_columns(df):
    for c in necessary_columns:
        if c not in df.columns:
            raise ValueError('The stock dataset is missing ' + str(c) + ' column')