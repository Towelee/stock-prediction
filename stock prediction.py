from pandas_datareader import DataReader
from datetime import datetime

#goog = DataReader('GOOG', 'yahoo', datetime(2020,4,1), datetime(2020,5,5))

#amzn = DataReader('AMZN', 'yahoo', datetime(2020,4,1), datetime(2020,5,5))

appl = DataReader('AAPL', 'yahoo', datetime(2010, 1, 1), datetime(2020, 6, 25))

