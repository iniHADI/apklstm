import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("Prediksi Inflasi Bulanan di Indonesia dengan LSTM")

# Load inflation data from Excel file
@st.cache_data
def load_data():
    df = pd.read_excel('Data Inflasi.xlsx')
    return df

data = load_data()

# Assuming the Excel file has columns 'Date' and 'Inflation'
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

st.subheader("Data Inflasi Bulanan")
st.line_chart(data)

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
inflation_scaled = scaler.fit_transform(data)

def create_dataset(dataset, look_back=3):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
X, y = create_dataset(inflation_scaled, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
st.subheader("Training Model LSTM")
if st.button("Mulai Training"):
    with st.spinner('Training model...'):
        model.fit(X, y, epochs=20, batch_size=1, verbose=0)
    st.success("Training selesai!")

    # Predict future inflation for next 12 months
    last_data = inflation_scaled[-look_back:]
    predictions = []
    current_input = last_data.reshape(1, look_back, 1)
    for _ in range(12):
        pred = model.predict(current_input)[0,0]
        predictions.append(pred)
        current_input = np.append(current_input[:,1:,:], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='M')
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Inflation': predictions.flatten()})
    pred_df.set_index('Date', inplace=True)

    st.subheader("Prediksi Inflasi Bulanan 12 Bulan ke Depan")
    st.line_chart(pred_df)

else:
    st.info("Klik tombol 'Mulai Training' untuk melatih model dan melihat prediksi.")
