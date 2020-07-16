from pandas_datareader import DataReader
from datetime import datetime
from datetime import date
import numpy as np 
import pandas as pd
import seaborn as sb 
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt


##### Read Data
company  = 'GOOG'
cutoffyear = 2017

dat = DataReader(company, 'yahoo', datetime(2009, 1, 1), datetime.today())

##### Split Train and Test
train, test = dat[dat.index.year <= cutoffyear ], dat[dat.index.year > cutoffyear]


##### Scale close prices based on MinMaxScaler on training set -> need to scale x and y separate to inverse transform predictions y later
scaler_x = MinMaxScaler(feature_range = (0,1)) 
scaler_y = MinMaxScaler(feature_range = (0,1))

# scale train
train_sc_x = pd.DataFrame(scaler_x.fit_transform(train.filter(['Volume', 'Close'])), index = train.index)
train_sc_x.columns = train.filter(['Volume', 'Close']).columns +  '_sc'

train_sc_y = pd.DataFrame(scaler_y.fit_transform(train.filter(['Close'])), index = train.index)
train_sc_y.columns = train.filter(['Close']).columns + '_sc'

train_all = pd.concat([train, train_sc_x, train_sc_y], axis = 1).assign(date = lambda x: x.index) # use for reconciling results later

# scale test (with train scaler)
test_sc_x = pd.DataFrame(scaler_x.transform(test.filter(['Volume', 'Close'])), index = test.index)
test_sc_x.columns = test.filter(['Volume', 'Close']).columns +  '_sc'

test_sc_y = pd.DataFrame(scaler_y.transform(test.filter(['Close'])), index = test.index)
test_sc_y.columns = test.filter(['Close']).columns + '_sc'

test_all = pd.concat([test, test_sc_x, test_sc_y], axis = 1).assign(date = lambda x: x.index) # use for reconciling results later


##### Create feature sequences (x) and targets (y) from train and test sets separately (predict 51st day with past 50 days, on 1 day increments)

# From Tensorflow Documentation on time series predicion
# takes time series(s) and returns feature sequences x as np.array(data), and target y (single step or sequence) as np.array(labels)
# dataset = x, target = y
# start_index = 0 (index to start sampling at)
# end_index = last index of dataset being sampled -> should be len(dataset) - target size to avoid not having enough data for complete target sample at the end
# history_size = size of feature sequence (x): 50
# target size = size of target sequence (y): 5
# step = steps to increment by for each x,y sample (1 to match create_dataset() function in other stock prediction attempts for this project)
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step: 
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

x_var = ['Volume_sc', 'Close_sc'] 
y_var = ['Close_sc']

# need to reset index, drop date index, and to_numpy array for function to work (due to array slicing)
train_x, train_y = multivariate_data(train_sc_x[x_var].reset_index(drop=True).to_numpy(), train_sc_y[y_var].reset_index(drop=True).to_numpy(), start_index = 0, end_index = None, history_size = 50, target_size = 5, step = 1)
test_x, test_y = multivariate_data(test_sc_x[x_var].reset_index(drop=True).to_numpy(), test_sc_y[y_var].reset_index(drop=True).to_numpy(), start_index = 0, end_index = None, history_size = 50, target_size = 5, step=1)

print ('Single window of past history : {}'.format(train_x[0].shape)) ## shapes look correct (n rows, 50 steps, 2 features)
print ('\n Target window to predict : {}'.format(train_y[0].shape)) ## wrong : should be (5,) (n rows,  5 steps, 1 label

## convert shape of target
train_y.shape = (train_y.shape[0], 5)
test_y.shape = (test_y.shape[0], 5)


print ('Single window of past history : {}'.format(test_x[0].shape)) ## shapes look correct (50 steps, 2 features)
print ('\n Target window to predict : {}'.format(test_y[0].shape)) ## check that shape conversion worked


## build model
def buildModel(dataLength, labelLength): # 50, 5 (predict 5 steps into the future)

    # define layers
    close_sc = tf.keras.Input(shape = (dataLength, 1), name = 'close_sc')
    volume_sc = tf.keras.Input(shape = (dataLength, 1), name = 'volume_sc')

    close_sc_layers = tf.keras.layers.LSTM(64, return_sequences = False)(close_sc)
    volume_sc_layers = tf.keras.layers.LSTM(64, return_sequences = False)(volume_sc)

    output = tf.keras.layers.concatenate(
        [
            close_sc_layers,
            volume_sc_layers,
        ]
    )
    output = tf.keras.layers.Dense(labelLength, activation = 'linear', name = 'weightedAverage_output')(output)

    #define model with layers
    model = tf.keras.Model(
        inputs = 
        [
            close_sc,
            volume_sc
        ],
        outputs = 
        [
            output
        ]
    )

    model.compile(optimizer = 'adam', loss = 'mse')

    return model

rnn = buildModel(train_x.shape[1], 5)

## train
logdir = "logs/Graphs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) # tensorboard --logdir logs/scalars3/

rnn.fit(
    [
        train_x[:, :, 1], # slice by 3rd dimension of train_x array (2nd column = index 1 = close_sc)
        train_x[:, :, 0] # index 0 = volume
    ], 
    [
        train_y
    ],
     
epochs = 1, batch_size = 32, 
callbacks = [tensorboard_callback], 
validation_split = 0.05, # using some data for validation split hurts test performance the most
shuffle = False  ### shuffle = True works better even though it's time series?? -> because of leaked info from future sequences
)

##### Predict Test Set
pred = rnn.predict(
    [
        test_x[:, :, 1],
        test_x[:, :, 0]
    ]
)
pred_unscale = scaler_y.inverse_transform(pred) # unscale with same scaler as training 

##### Results
pred_df = pd.DataFrame(pred_unscale)
pred_df.columns = ['pred1', 'pred2', 'pred3', 'pred4', 'pred5']

test_y_df = pd.DataFrame(scaler_y.inverse_transform(test_y.reshape(test_y.shape[0], test_y.shape[1])))
test_y_df.columns = ['truth1', 'truth2', 'truth3', 'truth4', 'truth5']
res = pd.concat([pred_df, test_y_df], axis = 1)


sb.lineplot(res.pred1, res.truth1)
sb.lineplot(data = pred_df.iloc[1])
sb.lineplot(data = test_y_df.iloc[1])
