import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger le fichier CSV
df = pd.read_csv("vibrations_pieces.csv")

features = ["RMS", "Kurtosis", "Skewness", "Peak2Peak"]
target = "label"

# Vérifier colonnes requises
if not all(f in df.columns for f in features + [target]):
    raise ValueError(f"Le fichier CSV doit contenir les colonnes : {features + [target]}")

X = df[features]
y = df[target]

# Encodage du label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Entraîner modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Sauvegarde
joblib.dump((model, label_encoder), "model.pkl")
print("✅ Modèle entraîné et sauvegardé sous 'model.pkl'")