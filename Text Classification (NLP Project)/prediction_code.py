import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

data       = pd.read_csv('testing_dataset.csv') # Upload data
model      = joblib.load('model.pkl')           # Upload model
vectorizer = joblib.load('vectorizer.pkl')      # Upload Vectorizer

print(data.head())
print(data.isnull().sum())
print(data.duplicated().sum())

# Text Preprocessing
data['cleaned_text'] = data['Text']
data['cleaned_text'] = data['cleaned_text'].str.lower()

# Remove special symbols (such as #, @)
data['cleaned_text'] = data['cleaned_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Divide texts into words
data['cleaned_text'] = data['cleaned_text'].apply(word_tokenize)

# Remove (StopWords)
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['cleaned_text'].apply(lambda words: [word for word in words if word not in stop_words])

# Apply PorterStemmer
stemmer = PorterStemmer()
data['cleaned_text'] = data['cleaned_text'].apply(lambda words: [stemmer.stem(word) for word in words])

# Combine words back into sentences
data['cleaned_text'] = data['cleaned_text'].apply(lambda words: ' '.join(words))

# CountVectorizer
x_text = vectorizer.transform(data['cleaned_text'])

prediction = model.predict(x_text)

print('Output',prediction)
