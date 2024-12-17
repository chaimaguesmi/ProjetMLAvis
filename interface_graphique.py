import tkinter as tk
from tkinter import messagebox
import joblib
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Fonction de prétraitement des textes
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Retirer les caractères spéciaux
    text = re.sub(r'\s+', ' ', text)  # Retirer les espaces multiples
    text = ' '.join(word for word in text.split() if word not in stopwords.words('french'))
    return text

# Charger les outils sauvegardés
rf_model = joblib.load("random_forest_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Fonction pour la prédiction
def predict_sentiment():
    commentaire = entry_commentaire.get("1.0", tk.END).strip()
    if not commentaire:
        messagebox.showwarning("Entrée vide", "Veuillez entrer un commentaire pour la prédiction.")
        return
    try:
        # Prétraiter le texte
        commentaire_propre = preprocess_text(commentaire)
        commentaire_tfidf = tfidf_vectorizer.transform([commentaire_propre])
        prediction_encoded = rf_model.predict(commentaire_tfidf)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        messagebox.showinfo("Résultat de la prédiction", f"Sentiment prédit : {prediction_label}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

# Création de l'interface graphique
root = tk.Tk()
root.title("Analyse des Avis - Random Forest")

# Label et zone de texte pour entrer le commentaire
label_instruction = tk.Label(root, text="Entrez un commentaire pour prédire le sentiment :", font=("Helvetica", 12))
label_instruction.pack(pady=10)

entry_commentaire = tk.Text(root, height=5, width=60)
entry_commentaire.pack(pady=10)

# Bouton pour lancer la prédiction
button_predict = tk.Button(root, text="Prédire le Sentiment", command=predict_sentiment, bg="blue", fg="white", font=("Helvetica", 12))
button_predict.pack(pady=10)

# Lancement de l'interface
root.mainloop()
