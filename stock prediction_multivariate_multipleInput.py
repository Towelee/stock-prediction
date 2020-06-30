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
timesteps = 50
def create_dataset(X, y, time_steps):
# takes training or test df -> returns feature sequences (x) and corresponding targets (y)
### use _sc dataframes 
    Xs = []
    ys = []

    for i in range(len(X) - time_steps): # range(start = 0, finish= n-1)
        vx = X.iloc[i: (i + time_steps)].to_numpy() # 1st x sequence is time 0 to time_step -1 (49 = 50-1): returns index 0 to 49
        #train_sc_x.iloc[0:(0 + 50)] # check x where i=0
        #train_sc_x_test = train_sc_x.reset_index()
        #train_sc_x_test.iloc[0:(0+50)]

        Xs.append(vx)

        vy = y.iloc[i + time_steps] # first y is time_step (50) iloc[n]: returns index n, iloc[n:m]: returns n to m-1
        #train_sc_y.iloc[0 + 50] # check y i=0
        #train_sc_y_test = train_sc_y.reset_index()  
        #train_sc_y_test.iloc[0:51]
        ys.append(vy)
        
    # checks look good
    return np.array(Xs), np.array(ys)

x_var = ['Volume_sc', 'Close_sc'] 
y_var = ['Close_sc']

train_x, train_y = create_dataset(train_sc_x[x_var], train_sc_y[y_var], timesteps)
test_x, test_y = create_dataset(test_sc_x[x_var], test_sc_y[y_var], timesteps)


## check that 1st x sequence = 0 to timestep
## check that 1st y element = timestep+1
train_x[0]
train_y[0]
train_x[1]

train_y[1]
train_x[2]

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

## build model
def buildModel(dataLength, labelLength): # 50, 1

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

rnn = buildModel(train_x.shape[1], 1)

## train
logdir = "logs/Graphs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) # tensorboard --logdir logs/scalars3/

rnn.fit(
    [
        train_x[:, :, 1], # slice by 3rd dimension of train_x array (2nd column = index 1 = close_sc)
        train_x[:, :, 0]
    ], 
    [
        train_y
    ],
     
epochs = 100, batch_size = 32, 
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
padded_test = pd.concat([pd.DataFrame(np.repeat(np.nan, 50)), pd.DataFrame(test_y)], axis = 0).reset_index(drop = True) # pad truth (test y) with 50 nan (first prediction is 51st day of test set)
padded_test.columns = ['test_y'] # change column name
padded_pred = pd.concat([pd.DataFrame(np.repeat(np.nan, 50)), pd.DataFrame(pred_unscale)], axis = 0).reset_index(drop = True)
padded_pred.columns = ['pred']
results= pd.concat([padded_test, padded_pred, test_all.set_index(padded_test.index)], axis = 1).set_index('date').assign(residual = lambda x: x.Close - x.pred) ## needed to reset index of test_sc to match padded_test (couldn't reset padded test b/c no set_index for Series)


##### Plot Results
sb.lineplot(data = results[['pred', 'Close']]).set_title(company)
#sb.lineplot(data = results[['residual']])





