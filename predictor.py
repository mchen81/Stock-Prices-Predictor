#!/usr/bin/python

# general 
import sys
import stock_helper
import pandas as pd
import numpy as np
import datetime
from business_calendar import Calendar, MO, TU, WE, TH, FR

# machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import xgboost as xgb


arg_length = len(sys.argv)

if arg_length < 2:
    raise ValueError("Missing argument: stock code")
elif arg_length == 2:
    timeperiod=5
    print("Time period is set to 5(default).")
else:
    timeperiod=int(sys.argv[2])
    print("Time period is set to " + str(timeperiod) + ".")

stock_code=sys.argv[1].upper()

print("Finding " + stock_code + " from Yahoo Finance ...")

# fetching data from yahoo finance
dataset = stock_helper.fetch_stock_data(stock_code)

print(stock_code + " is downloaded, start to train models ...")

# create datafram with indicators
df = stock_helper.generate_indicators(dataset, 
                                      timeperiod=timeperiod, 
                                      generate_target=True, 
                                      reset_index=True)

date_series = df['date']

# Convert date to the n-th day from 1970-01-01, and rename it to day
df['date'] = df['date'].apply(lambda date64: (date64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D'))
df = df.rename(columns={'date': 'day'})

# real_time_test is the last n days in this dataset, where n is timeperiod
real_time_test = df.iloc[-timeperiod:,:-1]

# romove the last n(timeperiod) from the dataset
df = df.iloc[:-timeperiod,:]

df = df.dropna()

# target nanme could be vary
target_name = df.columns[-1]

# Split the dataset

split_point = int(len(df) * 0.8)
X = df.iloc[:, :-1]
Y = df[target_name]

train_X = X.iloc[:split_point, :]
train_y = Y[:split_point]

test_X = X.iloc[split_point:, :]
test_y = Y[split_point:]


# Initialize models 
models = []
models.append(DecisionTreeClassifier(max_depth=5))
models.append(RandomForestClassifier(max_depth=5))
models.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5))
models.append(xgb.XGBClassifier(use_label_encoder=False, eval_metric='error'))
models.append(SVC())
models.append(KNeighborsClassifier())
models.append(LogisticRegression())
models.append(GaussianNB())

accuracies = [None] * len(models)

for i, model in enumerate(models):
    model.fit(train_X, train_y)
    hyp = model.predict(test_X)
    accuracies[i] = accuracy_score(test_y, hyp)
    
# Get the best model
max_accuracy_index = accuracies.index(max(accuracies))
max_accuracy_model = models[max_accuracy_index]

print("The best Model is " + str(max_accuracy_model).split('(')[0])
print("Accuracy: {:.2f}%".format(accuracies[max_accuracy_index] * 100))
print("Start predicting close prices in the following " + str(timeperiod) + " days.")

# Re-train the model with whole dataset, and make a prediction for the last n(timeperiod) days.
max_accuracy_model.fit(X, Y)
prediction = max_accuracy_model.predict(real_time_test)

print("=======================================================")

close_prices = real_time_test.close.to_numpy()
days = date_series[-timeperiod:].values.astype("datetime64[D]")
cal = Calendar()

# Stock market does not open during weekends, so we need to add n(timeperiod) business days 
# representing the future days
following_days = [cal.addbusdays(str(day), timeperiod) for day in days]

# print out the result
for i in range(timeperiod):
    comparator = ''
    if prediction[i] == 1:
        comparator = '>'
    else:
        comparator = '<'
        
    print("On {date}, {stock}'s close price would be {comparator} {price}" \
        .format(date=following_days[i].date(),
                stock=stock_code,
                comparator=comparator,
                price=close_prices[i])
         )


print("=======================================================")
print("GOOD LUCK!")