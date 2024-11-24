import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
import joblib

# Read data
data = pd.read_csv('training_dataset.csv',header=None)

data.columns = ['entity','sentiment','content']

print(data.head())                        
print(data.isnull().sum())
print(data.duplicated().sum())
print(data['sentiment'].value_counts())

#Drop duplicates values
data = data.drop_duplicates()

# filling the missing values with most frequent
most = data['content'].mode()[0]
data['content'] = data['content'].fillna(most)

# Text Preprocessing
data['cleaned_text'] = data['content']
data['cleaned_text'] = data['cleaned_text'].str.lower()

# Remove special symbols (such as #, @)
data['cleaned_text'] = data['cleaned_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Remove extra spaces
data['cleaned_text'] = data['cleaned_text'].str.strip()

# CountVectorizer
vectorizer = CountVectorizer()  #CountVectorizer
encoder = LabelEncoder()        #LabelEncoder

x_text = vectorizer.fit_transform(data['cleaned_text'])     #Vectorizer ['cleaned_text'] 

x_add = encoder.fit_transform(data['entity']).reshape(-1, 1) 

x_combined = hstack([x_add, x_text]) 

# Split data into [x] [y]
x = x_combined
y = data['sentiment']

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

#Save Model & LabelEncoder & CountVectorizer
joblib.dump(model,'model.pkl')           #The Model
joblib.dump(encoder,'LabelEncoder.pkl')  #The LabelEncoder
joblib.dump(vectorizer,'vectorizer.pkl') #The CountVectorizer