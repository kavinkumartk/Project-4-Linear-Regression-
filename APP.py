import streamlit as st
import pickle
import numpy as np


with open("movie_model.sav", "rb") as f:
    model = pickle.load(f)

st.title("🎬 Box Office Revenue Predictor")

st.write("Enter the **Marketing Spend (in Crores)** to predict the **Box Office Revenue (in Crores).**")

marketing_spend = st.number_input("Marketing Spend (Cr):", min_value=0, max_value=1000, value=50)

features = np.array([[marketing_spend]])

if st.button("Predict Revenue"):
    prediction = model.predict(features)
    st.success(f"Estimated Box Office Revenue: ₹{prediction[0]:,.2f} Cr")
