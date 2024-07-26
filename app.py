import pickle
import streamlit as st
import pickle as pickle
import numpy as np

# Membaca model
customer_model = pickle.load(open('model.sav','rb'))

# Judul web
st.title('Prediksi Harga Kendaraan')

# Input data dengan contoh angka valid untuk pengujian
year = st.text_input('year')
mileage = st.text_input('mileage')
tax = st.text_input('tax')
mpg = st.text_input('mpg')
engineSize = st.text_input('engineSize')

Prediksi_Harga_Kendaraan = ''

# Membuat tombol untuk prediksi
if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[float(year), float(mileage), float(tax), float(mpg),float(engineSize))]])
        # Lakukan prediksi
        status_prediksi = customer_model.predict(inputs)
        
