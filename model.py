import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Titre ---
st.title("📈 Prédiction linéaire avec modèle sauvegardé")

# --- Charger le modèle ---
theta = joblib.load("linear_model.pkl")
st.write(f"**Coefficients du modèle :** θ0 = {theta[0]:.2f}, θ1 = {theta[1]:.2f}")

# --- Entrée utilisateur ---
x_input = st.slider("Valeur de X", min_value=0, max_value=200, value=60)

# --- Calcul de la prédiction ---
x_vec = np.array([1, x_input])  # colonne de biais + X
y_pred = theta @ x_vec

st.success(f"✅ La valeur prédite Y pour X = {x_input} est : {y_pred:.2f}")

# --- Affichage graphique ---
st.subheader("Graphique de la régression")
plt.scatter([50,70,90,110,130], [15,22,28,35,42], color='blue', label='Données')
x_line = np.linspace(0, 150, 100)
y_line = theta[0] + theta[1]*x_line
plt.plot(x_line, y_line, color='red', label='Modèle')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
st.pyplot(plt)
