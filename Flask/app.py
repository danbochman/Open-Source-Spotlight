import pickle

import numpy as np
import tensorflow as tf
from flask import Flask, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


def load():
    global services
    services = dict()

    # load the Sentiment Analysis pre-trained models' weights
    services['sa_model'] = load_model('storage/model.h5')

    # load the fitted tokenizer instance
    with open('storage/tokenizer.pkl', 'rb') as f:
        services['tokenizer'] = pickle.load(f)

    # pre-load the TensorFlow graph
    services['graph'] = tf.get_default_graph()


@app.route('/', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        # get the raw string from the webpage
        tweet = request.form.get('tweet')

        # preprocess it to the SA model format
        tweet = pad_sequences(services['tokenizer'].texts_to_sequences([tweet]), maxlen=300)

        # predict sentiment
        with services['graph'].as_default():
            prediction = services['sa_model'].predict(np.array(tweet))

        return ('''
                <h1> Sentiment Analysis for Tweet is: {}% </h1> 
                <button> <a href='http://127.0.0.1:5000/'>Back</a> </button>
                '''.format(round(100 * prediction.squeeze(), 2)))

    return ('''
            <form method="post">
            Type a Tweet to analyze its sentiment: <br>
            Tweet: <input type="text" name="tweet"> <br>
            <input type="submit" value="Submit"> <br>
            </form>
            ''')


def main():
    load()
    app.run()


if __name__ == '__main__':
    main()
