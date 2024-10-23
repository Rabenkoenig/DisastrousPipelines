# train_classifier.py

import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle

# Import libraries for the machine learning pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# Tokenization function
url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

def tokenize(text):
    # Detect and replace URLs
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
def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("categorized_messages", engine)
    
    # Split into feature (X) and target (Y) variables
    X = df["message"]
    Y = df.drop(columns=["id", "message", "original", "genre"])
    
    # Get category names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

# Build a machine learning pipeline
def build_model():
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    ("vect", CountVectorizer(tokenizer=tokenize)),
                                    ("tfidf", TfidfTransformer()),
                                ]
                            ),
                        ),
                        ("starting_verb", StartingVerbExtractor()),
                    ]
                ),
            ),
            ("multi_rf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    # Define parameters for GridSearchCV
    parameters = {
        "features__text_pipeline__vect__ngram_range": [(1, 1), (1, 2)],
        "features__text_pipeline__vect__max_df": [0.5, 0.75, 1.0],
        "features__text_pipeline__tfidf__use_idf": [True, False],
        "multi_rf__estimator__n_estimators": [50, 100, 200],
        "multi_rf__estimator__max_depth": [None, 10, 20],
        "features__transformer_weights": [
            {"text_pipeline": 1, "starting_verb": 0.5},
            {"text_pipeline": 0.5, "starting_verb": 1},
        ],
    }

    # Return GridSearchCV object
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)
    
    return model

# Evaluate model performance
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f"Category: {col}\n")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("-" * 60)

# Save the model to a pickle file
def save_model(model, model_filepath):
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)

# Main function to orchestrate the process
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
