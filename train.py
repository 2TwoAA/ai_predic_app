import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Chargement des données
data = pd.read_csv("vibrations_pieces.csv")

# Séparation features / labels
X = data.drop("label", axis=1)
y = data["label"]

# Encodage simple des labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Rapport de classification
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Sauvegarde du modèle
joblib.dump((model, le), "model.pkl")
print("Modèle sauvegardé sous 'model.pkl'")
