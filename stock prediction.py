from pandas_datareader import DataReader
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error


##### Read Data

#goog = DataReader('GOOG', 'yahoo', datetime(2020,4,1), datetime(2020,5,5))
#amzn = DataReader('AMZN', 'yahoo', datetime(2020,4,1), datetime(2020,5,5))
appl = DataReader('AAPL', 'yahoo', datetime(2009, 1, 1), datetime(2020, 6, 25))
appl.assign(year = lambda x: x.index.year).filter(['Date', 'Close']) ## test for ttsplit function

##### Split Train and Test

def ttsplit(df, cutoffyear, target):
## takes stock info time series(s), cutoff year, and target choice -> returns train, test set based on cutoff year
    df1 = df.assign(year = lambda x: x.index.year).filter(['year', target]) # Date is x.index, '.year' extracts year from Date 

    return df1[df1.year <= cutoffyear], df1[df1.year > cutoffyear]

train, test = ttsplit(appl, 2017, 'Close')

##### Scale close prices based on MinMaxScaler on training set
scaler = MinMaxScaler(feature_range = (0, 1)) # need feature range (0,1) for transforming 1 column 'Close'

# scale train
train_sc = train.assign(close_scaled = scaler.fit_transform(train.filter(['Close']))) ##
train_sc_close = train_sc.close_scaled.values.reshape(-1, 1) ## reshape values of closed_scaled

# scale test (with train scaler)
test_sc = test.assign(close_scaled = scaler.transform(test.filter(['Close'])))
test_sc_close = test_sc.close_scaled.values.reshape(-1, 1)

##### Create feature sequences (x) and targets (y) from train and test sets separately (predict 51st day with past 50 days)
def create_dataset(df):
# takes training or test df -> returns feature sequences (x) and corresponding targets (y)
    x = []
    y = []

    for i in range(50, df.shape[0]):# shape[0] = rows
        x.append(df[i-50:i, 0]) ### need [, 0] to select element within array
        y.append(df[i, 0])
    
    x = np.array(x)
    y = np.array(y)

    return x, y

    train_x, train_y = create_dataset(train_sc_close)
    test_x, test_y = create_dataset(test_sc_close)

##### Reshape features (x) for LSTM Layer (need (_, _ , 1)) to make it 3D for RNN
train_x_reshape = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1)) ## shape = (nrow, ncol, 1) -> (records, num_features, 1) -> num_features  = length of feature sequence : 1 makes it 3D for RNN
test_x_reshape = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))



### Buld model
model = Sequential()
model.add(LSTM(units = 96, return_sequences = True, input_shape = (train_x_reshape.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 96, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 96, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 96))
model.add(Dropout(0.2))
model.add(Dense(units = 1)) # 

model.compile(loss = 'mean_squared_error', optimizer = 'adam')






