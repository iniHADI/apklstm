import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.title("Prediksi Inflasi Bulanan di Indonesia dengan LSTM")

# Load data Excel
@st.cache_data
def load_excel_data():
    df = pd.read_excel("Data Inflasi.xlsx", engine="openpyxl")
    df.rename(columns={'Periode': 'Periode', 'Data Inflasi': 'Inflation'}, inplace=True)
    df['Periode'] = pd.to_datetime(df['Periode'])
    df.set_index('Periode', inplace=True)
    return df

# Data inflasi resmi dari BI (Des 2024 - Mar 2025)
def get_bi_data():
    bi_df = pd.DataFrame({
        'Periode': pd.to_datetime(['2024-12-01', '2025-01-01', '2025-02-01', '2025-03-01']),
        'Inflation': [1.57, 0.76, -0.09, 1.03]
    })
    bi_df.set_index('Periode', inplace=True)
    return bi_df

# Gabungkan data Excel dan data BI
@st.cache_data
def load_combined_data():
    excel_data = load_excel_data()
    bi_data = get_bi_data()
    combined = pd.concat([excel_data, bi_data])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    return combined

data = load_combined_data()

st.subheader("Data Inflasi Bulanan Gabungan")
st.line_chart(data)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
inflation_scaled = scaler.fit_transform(data[['Inflation']])

# Siapkan data untuk LSTM
def create_dataset(dataset, look_back=3):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
X, y = create_dataset(inflation_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Bangun model LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

st.subheader("Training Model LSTM")
if st.button("Mulai Training"):
    with st.spinner("Training model..."):
        model.fit(X, y, epochs=20, batch_size=1, verbose=0)
    st.success("Training selesai!")

    # Prediksi 12 bulan ke depan
    last_data = inflation_scaled[-look_back:]
    predictions = []
    current_input = last_data.reshape(1, look_back, 1)

    for _ in range(12):
        pred = model.predict(current_input)[0, 0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    # Transformasi balik ke bentuk aslinya
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='M')
    pred_df = pd.DataFrame({'Periode': future_dates, 'Prediksi Inflasi': predictions.flatten()})
    pred_df.set_index('Periode', inplace=True)

    st.subheader("Prediksi Inflasi 12 Bulan ke Depan")
    st.line_chart(pred_df)

    st.dataframe(pred_df.style.format({'Prediksi Inflasi': '{:.2f}'}))

else:
    st.info("Klik tombol 'Mulai Training' untuk melatih model dan melihat prediksi.")
