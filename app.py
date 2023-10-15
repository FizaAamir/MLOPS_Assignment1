"""
Flask Application for Sentiment Analysis.
"""

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the IMDb tokenizer
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  # 10,000 most frequent words

tokenizer = Tokenizer(num_words=10000)
tokenizer.word_index = imdb.get_word_index()

# Load the trained model from the HDF5 file
model = load_model('sentiment_model.h5')

def predict_sentiment(text):
    """
    Predict sentiment for the given text.

    Args:
        text (str): The input text for sentiment prediction.

    Returns:
        str: The predicted sentiment (Positive/Negative).
    """
    max_length = 250  # Same as the model's max_length
    text = text.lower()
    text = text.split()
    text = [tokenizer.word_index.get(word, 0) for word in text]
    text = pad_sequences([text], maxlen=max_length)
    prediction = model.predict(text)
    return "Positive" if prediction >= 0.5 else "Negative"

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handle requests to the root URL.

    Returns:
        str: Rendered HTML template with prediction.
    """
    prediction = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input:
            prediction = predict_sentiment(user_input)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

