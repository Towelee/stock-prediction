from pandas_datareader import DataReader
from datetime import datetime
from datetime import date
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb 
plt.style.use('seaborn-whitegrid')

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error


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
train_sc_x = pd.DataFrame(scaler_x.fit_transform(train.filter(['High', 'Low', 'Open', 'Volume', 'Close'])), index = train.index)
train_sc_x.columns = train.filter(['High', 'Low', 'Open', 'Volume', 'Close']).columns +  '_sc'

train_sc_y = pd.DataFrame(scaler_y.fit_transform(train.filter(['Close'])), index = train.index)
train_sc_y.columns = train.filter(['Close']).columns + '_sc'

train_all = pd.concat([train, train_sc_x, train_sc_y], axis = 1).assign(date = lambda x: x.index) # use for reconciling results later

# scale test (with train scaler)
test_sc_x = pd.DataFrame(scaler_x.transform(test.filter(['High', 'Low', 'Open', 'Volume', 'Close'])), index = test.index)
test_sc_x.columns = test.filter(['High', 'Low', 'Open', 'Volume', 'Close']).columns +  '_sc'

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
        vx = X.iloc[i: (i + time_steps)].to_numpy() # 1st x sequence is time 0 to time_step
        Xs.append(vx)

        vy = y.iloc[i + time_steps] # first y is time_step      
        ys.append(vy)

    return np.array(Xs), np.array(ys)

x_var = ['High_sc', 'Low_sc', 'Open_sc', 'Volume_sc', 'Close_sc'] 
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

##### DONT NEED THIS BECAUSE WE HAVE MULTIPLE FEATUERS ALREADY: Reshape features (x) for LSTM Layer (need (_, _ , 1)) to make it 3D for RNN
#train_x_reshape = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1)) ## shape = (nrow, ncol, 1) -> (records, num_features, 1) -> num_features  = length of feature sequence : 1 makes it 3D for RNN
#test_x_reshape = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))



##### Buld model
model = Sequential()
model.add(LSTM(units = 96, return_sequences = True, input_shape = (train_x.shape[1], train_x.shape[2])))
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
logdir = "logs/scalars3/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(train_x, train_y, 
epochs = 50, batch_size = 32, 
callbacks = [tensorboard_callback], 
#validation_split = 0.1
#, shuffle = False  ### shuffle = True works better even though it's time series??
)

#predict
pred = model.predict(test_x)
pred_unscale = scaler_y.inverse_transform(pred) # unscale with same scaler as training 

##### Results
padded_test = pd.concat([pd.DataFrame(np.repeat(np.nan, 50)), pd.DataFrame(test_y)], axis = 0).reset_index(drop = True) # pad truth (test y) with 50 nan (first prediction is 51st day of test set)
padded_test.columns = ['test_y'] # change column name
padded_pred = pd.concat([pd.DataFrame(np.repeat(np.nan, 50)), pd.DataFrame(pred_unscale)], axis = 0).reset_index(drop = True)
padded_pred.columns = ['pred']
results= pd.concat([padded_test, padded_pred, test_all.set_index(padded_test.index)], axis = 1).set_index('date').assign(residual = lambda x: x.Close - x.pred) ## needed to reset index of test_sc to match padded_test (couldn't reset padded test b/c no set_index for Series)


##### Plot Results
sb.lineplot(data = results[['pred', 'Close']]).set_title(company)
sb.lineplot(data = results[['residual']])



