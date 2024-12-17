import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

nltk.download('stopwords')

# Charger les données
df = pd.read_csv("final_balanced_reviwes.csv", sep=";")

# Fonction de prétraitement des textes
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Retirer les caractères spéciaux
    text = re.sub(r'\s+', ' ', text)  # Retirer les espaces multiples
    text = ' '.join(word for word in text.split() if word not in stopwords.words('french'))
    return text

# Prétraiter les textes
df['Commentaire'] = df['Commentaire'].apply(preprocess_text)
X = df['Commentaire']
y = df['Label']

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoder les labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Vectorisation TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Optimisation des hyperparamètres Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train_encoded)

# Meilleur modèle
best_rf = grid_search.best_estimator_
print("Meilleurs hyperparamètres :", grid_search.best_params_)

# Fonction d'évaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title(f"Matrice de Confusion - {model_name}")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.show()

# Évaluer le modèle amélioré
evaluate_model(best_rf, X_test_tfidf, y_test_encoded, "Random Forest Optimisé")

# Comparer avec un autre modèle : Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train_tfidf, y_train_encoded)
evaluate_model(gb, X_test_tfidf, y_test_encoded, "Gradient Boosting")

# Sauvegarder les outils
joblib.dump(best_rf, "random_forest_model_optimized.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Modèle et outils sauvegardés avec succès !")
