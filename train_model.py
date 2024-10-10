from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from training_data import training_data  # Importation des données d'entraînement

# Extraction des phrases et des intentions depuis le training_data
train_sentences = [example for intent in training_data for example in intent['examples']]
train_labels = [intent['intent'] for intent in training_data for _ in intent['examples']]

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(train_sentences, train_labels, test_size=0.2, random_state=42)

# Vectorisation des phrases (conversion en vecteurs numériques)
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Modèle Naive Bayes pour la classification
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Évaluation du modèle
accuracy = model.score(X_test_vectors, y_test)
print(f"Précision du modèle: {accuracy:.2f}")
