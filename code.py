import numpy as np
import pandas as pd

np.random.seed(42)

# Distribution réaliste
n_normal = 8000
n_inner = 1433
n_outer = 567

def generate_class_data(label, size):
    if label == "Normal":
        rms = np.random.normal(0.01, 0.005, size)
        kurtosis = np.random.normal(3, 0.5, size)
        skewness = np.random.normal(0.1, 0.2, size)
        peak2peak = np.random.normal(0.04, 0.01, size)
    elif label == "Inner_Fault":
        rms = np.random.normal(0.04, 0.01, size)
        kurtosis = np.random.normal(7, 1.0, size)
        skewness = np.random.normal(1.0, 0.3, size)
        peak2peak = np.random.normal(0.15, 0.03, size)
    else:  # Outer_Fault
        rms = np.random.normal(0.035, 0.01, size)
        kurtosis = np.random.normal(7.5, 1.2, size)
        skewness = np.random.normal(0.9, 0.3, size)
        peak2peak = np.random.normal(0.12, 0.03, size)

    return pd.DataFrame({
        "RMS": rms,
        "Kurtosis": kurtosis,
        "Skewness": skewness,
        "Peak2Peak": peak2peak,
        "label": [label]*size
    })

# Génération des données
df_normal = generate_class_data("Normal", n_normal)
df_inner = generate_class_data("Inner_Fault", n_inner)
df_outer = generate_class_data("Outer_Fault", n_outer)

# Fusion et sauvegarde
df = pd.concat([df_normal, df_inner, df_outer], ignore_index=True)

# Supprimer ancien fichier si existe
import os
if os.path.exists("vibrations_pieces.csv"):
    os.remove("vibrations_pieces.csv")

df.to_csv("vibrations_pieces.csv", index=False)
print(f"✅ Dataset simulé créé avec {len(df)} échantillons.")
print(df['label'].value_counts())
