# CS663-Spring2021-Stock-Prices-Predictor

# Description

TODO

# Libaray
Before running the scripts, some external libraries have to be installed.   

1. [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html)
2. [yfinance](https://pypi.org/project/yfinance/)
3. [lxml](https://lxml.de/installation.html)
4. [business_calendar](https://pypi.org/project/business_calendar/)

**Other ML libraries like numpy, pandas, sklearn, tensorflow are also needed.**

# Prediction Script
The [predictor.py](/predictor.py) predicts the trend of next n days for a company.   
You **MUST** provide the first argument, which is stock code(can be found [here](https://finance.yahoo.com/lookup)).  
The second argument is time period, which is optional(default is 5 days).   
   
Valid arguments example: 
```
python3 predictor.py GOOG 

python3 predictor.py AAPL 10 
```

# Helper Script
The [stock_helper.py](/stock_helper.py) has two helper functions.
### 1. generate_indicators
This function generates other 7 indicators(as below) for provided stock data.   
```
RSI:  Relative Strength Index 
KD:   Stochastic Oscillator 
MOM:  Momentum 
MACD: Moving Average Convergence Divergence 
ADX:  Average Directional Indicator 
SMA:  Simple Moving Average 
BB:   Bollinger Bands (upper, middle and lower)
```

### 2. fetch_stock_data 
This method will fetch the real time stock data of a company(by stock_code)
Two paramters can be set:

* stock_code: Search for a company [here](https://finance.yahoo.com/lookup), then you will know its stock code. (e.g. Microsoft Corporation (MSFT))
* period: Accept units like year, month and day. For example, 1y=1 year , 2m=2 month, 10d=10 day. Fetch all data if it is not given.



# Notebook

TODO


