import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import joblib

# Load the trained model
model = load_model('spam_classifier_model.h5')

# Load the TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

# Assuming that I have a saved vectorizer
# vectorizer = joblib.load('vectorizer.pkl')

# Define the Streamlit UI
st.title('Spam Email Classifier')
st.write('Enter the text to classify whether it is spam or not:')
user_input = st.text_area('Input Text')

# Make predictions

if st.button('Predict'):
    if user_input.strip() == '':
        st.error('Please fill in the input field.')
    else:
        user_input_features = vectorizer.transform([user_input]).toarray()
        
        # user_input_features= vectorizer.transform(user_input) 
        prediction = model.predict(user_input_features)

        # Display results
        if prediction[0] > 0.5:
            st.success("Not a Spam Mail")
        else:
            st.success("Spam Mail")
