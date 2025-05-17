# Prédiction du diabète avec le Machine Learning

## Objectif de la WebApp

Cette application permet de prédire si une personne est atteinte de diabète en utilisant des données médicales. Le modèle utilise un algorithme de Machine Learning, plus précisément un **Random Forest Classifier**, pour faire la prédiction.

## Choix du Dataset

Le dataset utilisé pour ce projet est le **Pima Indians Diabetes Dataset**, qui contient des informations sur des patients, avec un attribut indiquant si chaque patient est atteint de diabète ou non. Ce dataset est souvent utilisé pour des démonstrations de modèles de prédiction de maladies.

- **Source du dataset** : [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Le dataset comprend plusieurs caractéristiques médicales, telles que :
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome 1 = Diabétique, 0 = Non diabétique

## Choix du Modèle

Le **Random Forest Classifier** est un modèle de prédiction fiable et fonctionne bien même avec un petit dataset. Il est capable de gérer des relations complexes entre les variables sans trop risquer de surapprentissage. Il donne également de bonnes performances de prédiction

## Fonctionnement de l'Application

1. **Chargement du Dataset** : L'application charge le dataset Pima Indians Diabetes.
2. **Entraînement du Modèle** : Le modèle de Random Forest est entraîné sur 80% des données, et les 20% restants sont utilisés pour tester la précision du modèle.
3. **Visualisation** : Un graphique affiche l’importance des différentes variables dans la prédiction du modèle. Cela aide à mieux comprendre les facteurs qui influencent le résultat.
3. **Prédiction** : L'utilisateur entre des informations médicales (telles que le nombre de grossesses, le taux de glucose, etc.) via une interface utilisateur simple.
4. **Affichage du Résultat** : L'application affiche la probabilité que la personne soit diabétique, et donne une réponse sous forme de succès ou d'erreur, selon la prédiction du modèle.
