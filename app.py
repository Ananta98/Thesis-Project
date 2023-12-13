import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import efficientnet.tfkeras as effnet
import altair as alt
import pandas as pd
from tensorflow_addons.metrics import F1Score

def prerprocess_image(file, img_height, img_width):
    image = Image.open(file)
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    image = tf.keras.preprocessing.image.img_to_array(rgb_image)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32)
    return image


st.title("COVID-19 X-Ray Detection App")

class_covid_detection = ['NORMAL', 'COVID-19', 'PNEUMONIA']


uploaded_image = st.file_uploader(
    "Upload File Chest X-Ray", type=["jpg", "jpeg", "png"])


def convert_to_dataframe(prediction_array, class_name):
    df = pd.DataFrame(
        {"Chest X-Ray Label": class_name, "Percentage": prediction_array}
    )
    return df


if uploaded_image is not None:
    keras.backend.clear_session()
    model = keras.models.load_model("D:/Skripsi/Final Skripsi/Model/EfficientNet-B1 Skripsi epoch 100.h5")
    
    image_show = Image.open(uploaded_image)
    st.image(image_show, use_column_width=True, clamp=True)
    normalization = keras.layers.experimental.preprocessing.Rescaling(1./255)

    image = prerprocess_image(uploaded_image, 240, 240)
    with st.spinner("Classifying:   "):
        
        image = tf.expand_dims(image, 0)
        image = normalization(image)
        predictions_covid19 = np.round(model.predict(image) * 100, 2)

        prediction_covid19_xray = class_covid_detection[np.argmax(
            predictions_covid19[0])]
        if prediction_covid19_xray == "COVID-19":
            st.markdown(
                f"<h2 style='text-align: center;color: red;'>Prediction : {prediction_covid19_xray}</h2>", unsafe_allow_html=True)
        elif prediction_covid19_xray == "NORMAL":
            st.markdown(
                f"<h2 style='text-align: center;color: green;'>Prediction : {prediction_covid19_xray}</h2>", unsafe_allow_html=True)
        elif prediction_covid19_xray == "PNEUMONIA":
            st.markdown(
                f"<h2 style='text-align: center;color: yello;'>Prediction : {prediction_covid19_xray}</h2>", unsafe_allow_html=True)

        data_covid19_xray = convert_to_dataframe(
            predictions_covid19[0], class_covid_detection)

        st.title("Pneumonia Statistics:")
        bars = alt.Chart(data_covid19_xray).mark_bar().encode(
            x='Chest X-Ray Label',
            y='Percentage'
        )
        st.altair_chart(bars, use_container_width=True)
