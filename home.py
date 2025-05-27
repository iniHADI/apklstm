import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("Prediksi Inflasi Bulanan Indonesia")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("Data Inflasi (2).xlsx", engine="openpyxl")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df.set_index("Tanggal", inplace=True)
    df = df[["Inflasi"]]  # hanya kolom inflasi
    return df

data = load_data()

st.subheader("Data Inflasi (2003–2025)")
st.line_chart(data)

# 2. Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, look_back=3):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(y)

look_back = 3
X, y = create_dataset(scaled_data, look_back)
X = X.reshape(X.shape[0], look_back, 1)

# 3. Model LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

# 4. Training
st.subheader("Training Model")
if st.button("Mulai Training"):
    with st.spinner("Sedang melatih model..."):
        model.fit(X, y, epochs=20, batch_size=1, verbose=0)
    st.success("Training selesai!")

    # 5. Prediksi bulan berikutnya
    last_input = scaled_data[-look_back:].reshape(1, look_back, 1)
    next_scaled = model.predict(last_input)
    next_inflation = scaler.inverse_transform(next_scaled)[0, 0]
    last_real = data["Inflasi"].iloc[-1]

    arah = "meningkat" if next_inflation > last_real else "menurun"

    # 6. Output
    st.subheader("Prediksi Bulan Berikutnya")
    st.write(f"Prediksi inflasi bulan depan: **{next_inflation:.2f}%**")
    st.write(f"Tingkat inflasi diperkirakan akan **{arah}** dibandingkan bulan terakhir (**{last_real:.2f}%**).")
else:
    st.info("Klik tombol 'Mulai Training' untuk melatih model dan melihat prediksi.")
