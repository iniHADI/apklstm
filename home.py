import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("Prediksi Inflasi Bulanan di Indonesia dengan LSTM")

# Load data from Excel
@st.cache_data
def load_data():
    df = pd.read_excel('Data Inflasi (2).xlsx', engine='openpyxl')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)
    df.rename(columns={'Inflasi': 'Inflation'}, inplace=True)
    return df

data = load_data()

st.subheader("Data Inflasi Bulanan")
st.line_chart(data)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
inflation_scaled = scaler.fit_transform(data)

# Prepare dataset for LSTM
def create_dataset(dataset, look_back=3):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
X, y = create_dataset(inflation_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape to [samples, time steps, features]

# Build model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

st.subheader("Training Model LSTM")
if st.button("Mulai Training"):
    with st.spinner("Training model..."):
        model.fit(X, y, epochs=20, batch_size=1, verbose=0)
    st.success("Training selesai!")

    # Predict 12 months ahead
    last_data = inflation_scaled[-look_back:]
    predictions = []
    current_input = last_data.reshape(1, look_back, 1)

    for _ in range(12):
        pred = model.predict(current_input)[0, 0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='M')
    pred_df = pd.DataFrame({'Tanggal': future_dates, 'Prediksi Inflasi': predictions.flatten()})
    pred_df.set_index('Tanggal', inplace=True)

    st.subheader("Prediksi Inflasi 12 Bulan ke Depan")
    st.line_chart(pred_df)

else:
    st.info("Klik tombol 'Mulai Training' untuk melatih model dan melihat prediksi.")
