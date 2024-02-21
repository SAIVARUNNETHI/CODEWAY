import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('/content/drive/MyDrive/spam.csv',encoding='latin1')


if 'v1' in df.columns and 'v2' in df.columns:
    X = df['v2']
    y = df['v1']
tfidf_vectorizer = TfidfVectorizer()


X_tfidf = tfidf_vectorizer.fit_transform(X)

nb_classifier = MultinomialNB()


nb_classifier.fit(X_tfidf, y)

y_pred = nb_classifier.predict(X_tfidf)
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
classification_rep = classification_report(y, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
