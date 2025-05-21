import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Streamlit App title
st.title("Stock Index Predictor Using LSTM")

# Streamlit sidebar inputs for the stock data
st.sidebar.header("Stock Data Inputs")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2023-01-01'))

# Fetch stock data
st.write(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...")
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display stock data in the app
st.subheader(f"Stock Data for {ticker_symbol}")
st.write(data.tail())

# Plot the closing price of the stock
st.subheader(f"Closing Price for {ticker_symbol}")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Close'])
ax.set_title(f"{ticker_symbol} Closing Price")
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
st.pyplot(fig)

# Data Preprocessing
st.subheader("Data Preprocessing")

# Use only 'Close' prices
close_prices = data['Close'].values
close_prices = close_prices.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Prepare training and test sets
training_data_len = int(np.ceil( len(scaled_data) * 0.8 ))

train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

# Prepare the data for training the LSTM model
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data for the LSTM input shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the LSTM model
st.subheader("Building the LSTM Model")

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
st.write("Training the model...")
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Predict the stock prices
st.subheader("Predicting the Stock Prices")

# Create the test data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = close_prices[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Reshaping the test data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Evaluate model performance
mse = mean_squared_error(y_test, predicted_prices)
st.write(f"Mean Squared Error: {mse}")

# Plot the results
st.subheader("Stock Price Prediction vs Actual")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(data['Close'].index[training_data_len:], y_test, color='blue', label='Actual Price')
ax2.plot(data['Close'].index[training_data_len:], predicted_prices, color='red', label='Predicted Price')
ax2.set_title(f"{ticker_symbol} Price Prediction")
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)
