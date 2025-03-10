# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb_data.h5')

# Step 2: Helper Functions
def predict_sentiment(text, model, word_index, max_len):
    # Convert text to sequence
    words = text.lower().split()
    seq = [word_index.get(word, 0) for word in words if word in word_index]
    seq = seq[:max_len]  # Truncate if too long
    seq = sequence.pad_sequences([seq], maxlen=max_len)
    
    # Predict
    prediction = model.predict(seq)[0][0]
    return 'Positive' if prediction >= 0.5 else 'Negative', prediction


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    ## MAke prediction
    sentiment, score = predict_sentiment(user_input, model, word_index, 200)

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score:.4f}')
else:
    st.write('Please enter a movie review.')

