import pandas as pd
import joblib
from scipy.sparse import hstack

data       = pd.read_csv('testing_dataset.csv') # Upload data
model      = joblib.load('model.pkl')           # Upload model
encoder    = joblib.load('LabelEncoder.pkl')    # Upload LabelEncoder
vectorizer = joblib.load('vectorizer.pkl')      # Upload Vectorizer

data.columns = ['entity','content']

print(data.head())
print(data.isnull().sum())
print(data.duplicated().sum())

# Text Preprocessing
data['cleaned_text'] = data['content']
data['cleaned_text'] = data['cleaned_text'].str.lower()

# Remove special symbols (such as #, @)
data['cleaned_text'] = data['cleaned_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Remove extra spaces
data['cleaned_text'] = data['cleaned_text'].str.strip()

# CountVectorizer
x_text = vectorizer.transform(data['cleaned_text'])

x_add = encoder.transform(data['entity']).reshape(-1, 1)

x_combined = hstack([x_add, x_text])

prediction = model.predict(x_combined)

print('Output',prediction)