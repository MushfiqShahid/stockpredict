import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Load and preprocess data
df = pd.read_csv("C:/Users/hp/Desktop/stock_predict/data/alphabet.csv")
df.rename(columns={'Close/Last': 'Close'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

# Create sequences
SEQ_LEN = 60
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LEN)

# Split into training set (use all data for training)
X_train, y_train = X, y

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Get user input for prediction
last_date = df.index[-1]
print(f"Last available date in dataset: {last_date.date()}")
target_date_str = input("Enter a future date to predict stock price (YYYY-MM-DD): ")
target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

# Validate
if target_date <= last_date:
    print("Target date must be after the last date in the dataset.")
    exit()

# Start prediction from last known sequence
future_days = (target_date - last_date).days
input_sequence = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
predicted_scaled = []

for _ in range(future_days):
    pred = model.predict(input_sequence)[0]
    predicted_scaled.append(pred)
    input_sequence = np.append(input_sequence[:, 1:, :], [[pred]], axis=1)

# Inverse scale predictions
predicted_prices = scaler.inverse_transform(predicted_scaled)

# Show result
predicted_price_on_target = predicted_prices[-1][0]
print(f"Predicted price on {target_date.date()} is: ${predicted_price_on_target:.2f}")

# Plot forecast
dates = [last_date + timedelta(days=i+1) for i in range(future_days)]

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Historical Price')
plt.plot(dates, predicted_prices, label='Forecasted Price', color='orange')
plt.axvline(x=target_date, color='red', linestyle='--', label='Target Date')
plt.title('ALPHABET Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()