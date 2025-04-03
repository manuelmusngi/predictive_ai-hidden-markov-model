## Data Preprocessing

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the time series data
data = pd.read_csv('path_to_your_time_series_data.csv')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for RNN/LSTM
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features] for RNN/LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

## mplementing RNN and LSTM Models
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Create the RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(time_step, 1)))
rnn_model.add(SimpleRNN(units=50))
rnn_model.add(Dense(units=1))

# Compile the model
rnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
rnn_model.fit(X, y, epochs=100, batch_size=32, verbose=1)

from keras.models import Sequential
from keras.layers import LSTM, Dense

# Create the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X, y, epochs=100, batch_size=32, verbose=1)

## Evaluation and Visualization
import matplotlib.pyplot as plt

# Predict using RNN model
rnn_predictions = rnn_model.predict(X)
rnn_predictions = scaler.inverse_transform(rnn_predictions)

# Predict using LSTM model
lstm_predictions = lstm_model.predict(X)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(data, label='True Data')
plt.plot(range(time_step, len(rnn_predictions) + time_step), rnn_predictions, label='RNN Predictions')
plt.plot(range(time_step, len(lstm_predictions) + time_step), lstm_predictions, label='LSTM Predictions')
plt.legend()
plt.show()


