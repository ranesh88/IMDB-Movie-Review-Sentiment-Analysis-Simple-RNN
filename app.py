from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Helper function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the user input from the form
    review = request.form['review']

    # Preprocess the review
    preprocessed_input = preprocess_text(review)

    # Make a prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    score = prediction[0][0]

    # Render results on the HTML page
    return render_template('index.html', sentiment=sentiment, score=score, review=review)

if __name__ == '__main__':
    app.run(debug=True)
