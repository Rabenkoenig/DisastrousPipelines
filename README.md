# DisastrousPipelines

This project is a web-based application that classifies messages related to disaster response into multiple categories. It uses machine learning models to help identify the type of help or action required, such as requests for medical assistance, infrastructure damage, and other relevant categories.

## Table of Contents
- [Project Motivation](#project-motivation)
- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [Instructions](#instructions)
- [Web Application Features](#web-application-features)
- [Data Visualizations](#data-visualizations)
- [Acknowledgments](#acknowledgments)

## Project Motivation

The aim of this project is to build a tool to classify disaster messages. This allows for efficient filtering of critical information and supports timely response during disaster scenarios. This project is part of the Data Science Nanodegree Program, and it utilizes data from [Figure Eight](https://www.figure-eight.com) that includes messages sent during real-life disaster events.

## Installation

To install and run the application locally, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/disaster-response-app.git
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   Open a Python shell and run the following commands:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   ```

4. Ensure you have SQLite installed, as the app uses a database for disaster message categorization.

## File Descriptions

- `app/templates/`: Contains HTML templates for the web application (`master.html`, `go.html`).
- `app/run.py`: The Flask application file that sets up and runs the server.
- `data/process_data.py`: ETL pipeline to process messages and store them in a SQLite database.
- `models/train_classifier.py`: Script to train the machine learning model.
- `data/DisasterResponse.db`: SQLite database storing categorized disaster messages.
- `models/modelStartingVerb.pkl`: Trained machine learning model.
- `requirements.txt`: Contains the dependencies required to run the project.

## Instructions

1. **ETL Pipeline:**
   - To process the data and create the SQLite database, run:
     ```bash
     python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
     ```

2. **Model Training:**
   - To train the model and save it as a pickle file, run:
     ```bash
     python models/train_classifier.py ../data/DisasterResponse.db modelStartingVerb.pkl
     ```

3. **Run the Web App:**
   - After processing the data and training the model, run the Flask app using:
     ```bash
     python app/run.py
     ```
   - Navigate to `http://0.0.0.0:3000/` in your browser.

## Web Application Features

Once the application is running, it provides a user interface where users can input a message related to disaster response, and the app will categorize the message into multiple relevant categories.

## Data Visualizations

Three visualizations are provided in the web app:

1. **Distribution of Message Genres**: A bar chart showing the distribution of message genres (e.g., social, news).
2. **Top Categories Distribution**: A bar chart displaying the most common categories from the disaster dataset.
3. **Messages per Category**: A chart showing the frequency of messages per disaster response category, helping understand the volume and common types of messages.

## Acknowledgments

This project was completed as part of the [Udacity Data Science Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). The dataset was provided by [Figure Eight](https://www.figure-eight.com), and some of the code structure was adapted from Udacity's project templates.

