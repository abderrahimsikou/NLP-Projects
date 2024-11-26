# Text Classification (NLP Project)

A project to analyze and classify text into categories based on their content using,
Natural Language Processing (NLP) techniques and Machine Learning algorithm.
The goal of the project is to train a model to classify text into categories: 0,1,2,3,4

Politics = 0
Sport = 1
Technology = 2
Entertainment =3
Business = 4. 

# Features
- Clean and process text data using 'NLTK'.
- Extract features using 'CountVectorizer'.
- Use Naive Bayes algorithm for classification.
- Save model and CountVectorizer for reuse with new data.

# Requirements
Before running the project, make sure you have the following packages installed:

- Python 3.x
- Pandas
- Scikit-learn
- Joblib
- nltk (You need to download 'word_tokenize', 'stopwords')

How to download:
import ntlk
nltk.download('word_tokenize')
nltk.download('stopwords')

## Work steps

1. Train the model

Training file: training_dataset.csv

Contains columns:

Label: Label:It contains labels for five different categories : 0,1,2,3,4

Politics = 0
Sport = 1
Technology = 2
Entertainment =3
Business = 4

Text: texts to be classified.

Code:

Text cleaning:

Convert texts to lowercase.

Remove special characters (e.g. #, @).

Divide texts into words by using word_tokenize

Remove (StopWords)

Apply PorterStemmer

Extract features using CountVectorizer.

Split data into training and testing set.

Train the model using Naive Bayes algorithm.

Save the model and tools (Model & CountVectorizer).

# Files

1. training_model.py: Model training code.

2. prediction_code.py: New data prediction code.

3. training_dataset.csv: Training data.

4. testing_dataset.csv: Prediction data.

5. model.pkl: Saved model.

6. vectorizer.pkl: CountVectorizer tool.

# Note:
The model achieve accuracy up to 97%.