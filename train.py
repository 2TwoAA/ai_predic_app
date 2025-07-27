import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger le fichier CSV
df = pd.read_csv("vibrations_pieces.csv")

# Définir les features utilisées pour l'entraînement
features = ["RMS", "Kurtosis", "Skewness", "Peak2Peak"]

# Vérification que toutes les colonnes existent
if not all(f in df.columns for f in features + ["label"]):
    raise ValueError(f"Le fichier CSV doit contenir les colonnes : {features + ['label']}")

# Séparation des variables
X = df[features]
y = df["label"]

# Encodage de la cible (Label)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Sauvegarder le modèle et l'encodeur ensemble
joblib.dump((model, label_encoder), "model.pkl")

print("✅ Modèle entraîné et sauvegardé sous 'model.pkl'")
