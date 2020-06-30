from pandas_datareader import DataReader
from datetime import datetime
from datetime import date
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb 
plt.style.use('seaborn-whitegrid')
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error


##### Read Data
company  = 'GOOG'
#goog = DataReader('GOOG', 'yahoo', datetime(2020,4,1), datetime(2020,5,5))
#amzn = DataReader('AMZN', 'yahoo', datetime(2020,4,1), datetime(2020,5,5))
dat = DataReader(company, 'yahoo', datetime(2009, 1, 1), datetime.today())
dat.assign(year = lambda x: x.index.year).filter(['Date', 'Close']) ## test for ttsplit function

##### Split Train and Test

def ttsplit(df, cutoffyear, target):
## takes stock info time series(s), cutoff year, and target choice -> returns train, test set based on cutoff year
    df1 = df.assign(year = lambda x: x.index.year).filter(['year', target]) # Date is x.index, '.year' extracts year from Date 

    return df1[df1.year <= cutoffyear], df1[df1.year > cutoffyear]

train, test = ttsplit(dat, 2017, 'Close')

##### Scale close prices based on MinMaxScaler on training set
scaler = MinMaxScaler(feature_range = (0, 1)) # need feature range (0,1) for transforming 1 column 'Close' to range (0,1)
scaler2 = MinMaxScaler(feature_range = (0,1) )


# scale train
train_sc = train.assign(close_scaled = scaler.fit_transform(train.filter(['Close']))) ##
train_sc_close = train_sc.close_scaled.values.reshape(-1, 1) ## reshape values of closed_scaled

# scale test (with train scaler)
test_sc = test.assign(close_scaled = scaler.transform(test.filter(['Close'])), 
                      date = lambda x: x.index) ### copy date from index as a column
test_sc_close = test_sc.close_scaled.values.reshape(-1, 1)

# check scaler behavior in assign() -- same sc_v1 = sc
train_sc_v1 = scaler2.fit_transform(train.filter(['Close']))
test_sc_v1 = scaler2.transform(test.filter(['Close']))

##### Create feature sequences (x) and targets (y) from train and test sets separately (predict 51st day with past 50 days, on 1 day increments)
def create_dataset(df):
# takes training or test df -> returns feature sequences (x) and corresponding targets (y)
    x = []
    y = []

    for i in range(50, df.shape[0]):# shape[0] = rows
        x.append(df[i-50:i, 0]) ### need [, 0] to select element within array
        y.append(df[i, 0]) #### *** should this be i+1? 51st element is the first y? (NO: checked last element of x[0] != y[0])
    
    x = np.array(x)
    y = np.array(y)

    return x, y

train_x, train_y = create_dataset(train_sc_close)
test_x, test_y = create_dataset(test_sc_close)

## check that 1st x sequence = 0 to timestep
## check that 1st y element = timestep+1
train_x[0]
train_y[0]
train_x[1]

train_y[1]
train_x[2]

##### Reshape features (x) for LSTM Layer (need (_, _ , 1)) to make it 3D for RNN
train_x_reshape = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1)) ## shape = (nrow, ncol, 1) -> (records, num_features, 1) -> num_features  = length of feature sequence : 1 makes it 3D for RNN
test_x_reshape = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))



##### Buld model
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

#train
logdir = "logs/scalars5/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(train_x_reshape, train_y, 
epochs = 50, batch_size = 32, 
#validation_split = 0.1, 
#shuffle = False,
callbacks = [tensorboard_callback])

# $tensorboard --logdir logs/scalars5/ in terminal to get localhost link to tensorboard

#predict
pred = model.predict(test_x_reshape)
pred_unscale = scaler.inverse_transform(pred) # unscale with same scaler as training 

# check scaler behavior
pred_unscale_v1 = scaler2.inverse_transform(pred)
##### Results
#check that 51st closed_scale in test_sc == 1st test_y
test_sc[50:51] 
test_y[:1]

padded_test = pd.concat([pd.DataFrame(np.repeat(np.nan, 50)), pd.DataFrame(test_y)], axis = 0).reset_index(drop = True) # pad truth (test y) with 50 nan (first prediction is 51st day of test set)
padded_test.columns = ['test_y'] # change column name
padded_pred = pd.concat([pd.DataFrame(np.repeat(np.nan, 50)), pd.DataFrame(pred_unscale)], axis = 0).reset_index(drop = True)
padded_pred.columns = ['pred']
results= pd.concat([padded_test, padded_pred, test_sc.set_index(padded_test.index)], axis = 1).set_index('date').assign(residual = lambda x: x.Close - x.pred) ## needed to reset index of test_sc to match padded_test (couldn't reset padded test b/c no set_index for Series)


##### Plot Results
plt.figure(figsize = (8, 6))
sb.lineplot(data = results.query('year >=2016')[['pred', 'Close']]).set_title(company) # 'residual'
sb.lineplot(data = results[['residual']])



