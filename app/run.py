import json
import plotly
import pandas as pd
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from collections import Counter


app = Flask(__name__, template_folder="template")


# Tokenization function
url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

def tokenize(text):
    # Detect and replace URL
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens


# Custom transformer for extracting the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ["VB", "VBP"] or first_word == "RT":
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# Load data from the database
engine = create_engine("sqlite:///../data/processed/DisasterResponse.db")
df = pd.read_sql_table("DisasterResponse", engine)

# Load the model from the pickle file
model = joblib.load("../models/modelStartingVerb.pkl")


# Index webpage: Displays visuals and receives user input for classification
@app.route("/")
@app.route("/index")
def index():

    # TODO: Modify to extract data for your own visuals
    # Extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # TODO: Create your own visuals here
    # Example: Distribution of message categories (e.g., weather, aid, etc.)
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Tokenize the messages and count word frequencies
    df["tokenized_messages"] = df["message"].apply(tokenize)
    all_words = [word for tokens in df["tokenized_messages"] for word in tokens]
    word_counts = Counter(all_words)

    # Get the top 10 most common words
    top_words = word_counts.most_common(10)
    top_words_names = [word[0] for word in top_words]
    top_words_counts = [word[1] for word in top_words]

    # Create visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Bar(x=category_names, y=category_counts)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category", "tickangle": -45},
            },
        },
        {
            "data": [Bar(x=top_words_names, y=top_words_counts)],
            "layout": {
                "title": "Top 10 Most Frequent Words in Messages",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Words"},
            },
        },
    ]

    # Encode Plotly graphs in JSON format
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with Plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route("/go")
def go():
    # Save user input in query
    query = request.args.get("query", "")

    # Use the model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html. Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3000, debug=True)


if __name__ == "__main__":
    main()
