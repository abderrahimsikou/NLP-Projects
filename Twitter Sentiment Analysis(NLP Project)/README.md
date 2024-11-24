# Twitter Sentiment Analysis(NLP Project)

A simple project to analyze texts and classify them into categories based on their content using,
Natural Language Processing (NLP) techniques and Machine Learning algorithm.
The goal of the project is to train a classification model and use it for prediction on new data. 

# Features
- Clean and process text data.
- Extract features using 'CountVectorizer'.
- Use Naive Bayes algorithm for classification.
- Save model and tools for reuse with new data.

# Requirements
Before running the project, make sure you have the following packages installed:

- Python 3.x
- Pandas
- Scikit-learn
- Joblib
- Scipy

## Work steps

1. Train the model

Training file: training_dataset.csv

Contains columns:

entity: entity name (e.g. product or service).

content: texts to be classified.

sentiment: category (positive, negative, neutral, irrelevant).

Code:

Text cleaning:

Convert texts to lowercase.

Remove special characters (e.g. #, @).

Remove extra spaces.

Extract features using CountVectorizer.

Merge entities with textual features.

Split data into training and testing set.

Train the model using Naive Bayes algorithm.

Save the model and tools (CountVectorizer and LabelEncoder).

# Files

1. training_model.py: Model training code.

2. prediction_code.py: New data prediction code.

3. training_dataset.csv: Training data.

4. testing_dataset.csv: Prediction data.

5. model.pkl: Saved model.

6. vectorizer.pkl: CountVectorizer tool.

7. encoder.pkl: LabelEncoder tool
