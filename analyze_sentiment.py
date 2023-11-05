import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment_scores = sid.polarity_scores(text)

    # Determine the sentiment label based on the compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return sentiment

# Example usage
text = "I really enjoyed the movie. The acting was great!"
sentiment = analyze_sentiment(text)
print(sentiment)
