import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Replace these file paths with your actual file paths
train_data = load_data('/content/drive/MyDrive/train_data.txt')
test_data = load_data('/content/drive/MyDrive/test_data.txt')
test_data_solution = load_data('/content/drive/MyDrive/test_data_solution.txt')
descriptions = load_data('/content/drive/MyDrive/description.txt')

# Ensure all arrays have the same length
min_length = min(len(train_data), len(test_data), len(test_data_solution), len(descriptions))
train_data = train_data[:min_length]
test_data = test_data[:min_length]
test_data_solution = test_data_solution[:min_length]
descriptions = descriptions[:min_length]

df_train = pd.DataFrame({'description': descriptions, 'plot_summary': train_data})
df_test = pd.DataFrame({'description': descriptions, 'plot_summary': test_data})
df_test_solution = pd.DataFrame({'genre': test_data_solution})

# Assuming your dataset has 'plot_summary' and 'genre' columns
X_train = df_train['plot_summary']
y_train = df_test_solution['genre']

X_test = df_test['plot_summary']
y_test = df_test_solution['genre']

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


classifier = LogisticRegression(max_iter=500)
classifier.fit(X_train_tfidf, y_train)

movie_name = input("Enter the movie name: ")

# Make a prediction for the entered movie name
new_plot_summary = [movie_name] 
new_tfidf = tfidf_vectorizer.transform(new_plot_summary)
predicted_genre = classifier.predict(new_tfidf)[0]


print(f'Predicted Genre for "{movie_name}": {predicted_genre}')