import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Prédiction du diabète")

# Image d'en tête
image = Image.open("image.jpeg")
st.image(image, use_container_width=True)

# Titre et introduction
st.title("Prédiction du diabète à partir du jeu de données Pima Indians Diabetes.")
st.markdown("""
Bienvenue sur notre application de prédiction du diabète qui utilise un modèle de forêt aléatoire pour estimer la probabilité qu’une personne soit atteinte de diabète, en se basant sur les données médicales fournies.
""")

# Chargement des données
df = pd.read_csv("diabetes.csv")

# Aperçu du dataset
if st.checkbox("Afficher un aperçu des données"):
    st.write(df.head())

# Préparation des données
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
modele = RandomForestClassifier()
modele.fit(X_train, y_train)

# Évaluation du modèle
predictions = modele.predict(X_test)
precision = accuracy_score(y_test, predictions)
st.write(f"Précision du modèle : {precision * 100:.2f}%")

# Affichage de l’importance des variables
st.markdown("## Importance des variables utilisées dans le modèle")

importances = modele.feature_importances_
colonnes = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.title("Variables les plus importantes")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [colonnes[i] for i in indices])
plt.xlabel("Importance")
st.pyplot(plt)

# Formulaire pour faire une prédiction
st.markdown("## Prédiction personnalisée")

pregnancies = st.number_input("Nombre de grossesses", 0, 20, 1)
glucose = st.slider("Taux de glucose", 0, 200, 100)
blood_pressure = st.slider("Pression artérielle", 0, 140, 70)
skin_thickness = st.slider("Épaisseur de la peau", 0, 100, 20)
insulin = st.slider("Taux d’insuline", 0, 900, 80)
bmi = st.slider("Indice de masse corporelle (IMC)", 0.0, 70.0, 25.0)
dpf = st.slider("Antécédents familiaux (DPF)", 0.0, 2.5, 0.5)
age = st.slider("Âge", 10, 100, 33)

if st.button("Prédire"):
    donnees = [pregnancies, glucose, blood_pressure, skin_thickness,
               insulin, bmi, dpf, age]
    resultat = modele.predict([donnees])[0]

    if resultat == 1:
        st.error("D'après les données saisies, la personne a un risque élevé de diabète.")
    else:
        st.success("D'après les données saisies, la personne a un risque faible de diabète.")

# À propos
st.markdown("---")
st.subheader("À propos ")
st.markdown("""

Les résultats affichés sont donnés à titre indicatif et ne remplacent en aucun cas un diagnostic médical.

Réalisé par : Bintou DIOP et Mahatsindry Solo  
Date : 7 mai 2025
""")