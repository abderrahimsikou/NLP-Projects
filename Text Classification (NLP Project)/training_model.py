import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Read dataset
data = pd.read_csv('training_dataset.csv')

# Check dataset information
print(data.head())
print(data.isnull().sum())
print(data.duplicated().sum())
print(data['Label'].value_counts())

# Drop Duplicates values
data = data.drop_duplicates()

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
vectorizer = CountVectorizer()

x_text = vectorizer.fit_transform(data['cleaned_text']) # Vectorizer ['cleaned_text'] 

# Split data into [x] [y]
x = x_text
y = data['Label']

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.3,random_state=42)

#Training the model
model = MultinomialNB()
model.fit(x_train,y_train)

#Show Results of the model {accuracy}
prediction = model.predict(x_test)

accuarcy = accuracy_score(prediction,y_test)
print('accuracy_score',accuarcy * 100, '%')

cm = confusion_matrix(prediction,y_test)
print('cm:\n', cm)

classification_report = classification_report(prediction, y_test)
print('classification_report:\n' , classification_report)

# Save Model & CountVectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')