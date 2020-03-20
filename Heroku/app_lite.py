import nltk
from flask import Flask, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

app = Flask(__name__)

# Set the secret key
app.secret_key = b'your_secret_key'


def load():
    global services
    services = dict()

    # :oad the sentiment analysis lexicon data
    nltk.download('vader_lexicon')

    # initialize the VADER sentiment intensity analyzer
    services['sia_model'] = SentimentIntensityAnalyzer()


@app.route('/', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        # get the raw string from the webpage
        tweet = request.form.get('tweet')

        # predict sentiment from tweet
        sentiment = services['sia_model'].polarity_scores(tweet)['compound']

        return ('''
                <h1> Sentiment Analysis for Tweet is: {}</h1> 
                <button> <a href='#' onclick='history.go(-1)'> Back </a> </button>
                '''.format(round(sentiment, 2)))

    return ('''
            <form method="post">
            Type a Tweet to analyze its sentiment: <br>
            Tweet: <input type="text" name="tweet"> <br>
            <input type="submit" value="Submit"> <br>
            </form>
            ''')


def main():
    load()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
