import json
import re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Análisis de Sentimientos - RNN/LSTM", layout="centered")

MAX_LEN = 150

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("modelo_lstm.keras")
    with open("word_index.json", "r", encoding="utf-8") as f:
        word_index = json.load(f)
    return model, word_index

try:
    model, word_index = load_assets()
except Exception as e:
    st.error(f"Error al cargar archivos del modelo: {e}. Asegúrate de tener 'modelo_lstm.keras' y 'word_index.json' en la misma carpeta.")
    st.stop()

def preprocess_text(text, word_dict, max_len=150):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    
    encoded = [1]
    for w in words:
        encoded.append(word_dict.get(w, 2))
        
    return pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')

st.title("Clasificador de Sentimientos con LSTM (RNN)")
st.write("Escribe un comentario o reseña (en inglés preferiblemente, acorde al dataset de IMDb) para evaluar su polaridad:")

user_input = st.text_area("Ingresa tu reseña aquí:", height=130, 
                          value="This movie was absolutely wonderful, great acting and thrilling plot!")

if st.button("Analizar Sentimiento"):
    if not user_input.strip():
        st.warning("Por favor, ingresa un texto antes de presionar el botón.")
    else:
        with st.spinner("Procesando texto con la red LSTM..."):
            sequence_pad = preprocess_text(user_input, word_index, MAX_LEN)
            score = float(model.predict(sequence_pad)[0][0])
            
            is_positive = score >= 0.5
            sentiment = "Positivo" if is_positive else "Negativo"
            confidence = (score if is_positive else (1 - score)) * 100

        if is_positive:
            st.success(f"**Resultado:** Sentimiento {sentiment}")
        else:
            st.error(f"**Resultado:** Sentimiento {sentiment}")

        st.metric(label="Nivel de Confianza", value=f"{confidence:.2f}%")
        
        st.progress(score if is_positive else (1 - score))
        st.caption("Score crudo (0.0 = Muy negativo, 1.0 = Muy positivo): {:.4f}".format(score))