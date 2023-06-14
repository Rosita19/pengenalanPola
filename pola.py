import streamlit as st
import joblib
from PIL import Image

image = Image.open('wa.png')

st.image(image, use_column_width=True)
st.title("Analisis Sentimen Ulasan Aplikasi Whatsapp di Google Play Store Menggunakan Metode Naive Bayes")
st.write("""
    ##### Anggota Kelompok:\n
    Rosita Dewi Lutfiyah          200411100002 \n
    Arusal Khofiqoyni             200411100071
     """)
st.write("### Natural Language Processing Project for Final Exam")

# load model
model = joblib.load('model/bestmodelNB .pkl')
# load vectorize
vectorizer = joblib.load('model/tfidf_result  .pkl')

# inputan
ulasan = st.text_area('Masukkan ulasan')
button = st.button('Predict')

if button:
    # pembobotan menggunakan vectorize
    x_new = vectorizer.transform([ulasan])

    # predict menggunakan model
    predictions = model.predict(x_new)
    sentimen_class = ['Negatif', 'Netral', 'Positif']
    for i in predictions:
        st.write("Sentimen: ", sentimen_class[i])