# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io

st.title("Scoliosis Detection App")

uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the file to bytes
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    # Send the image to the FastAPI backend
    files = {'file': ('image.jpg', byte_im, 'image/jpeg')}
    response = requests.post("https://scoliosis-3.onrender.com/predict", files=files, data={'model_type': "all"})

    if response.status_code == 200:
        results = response.json()
        for model, prediction in results.items():
            st.write(f"{model.capitalize()}: Scoliosis detected - {prediction}")
    else:
        st.write("Error in prediction")

st.write("Note: This app is for educational purposes only and should not be used for medical diagnosis.")
