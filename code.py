import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 1000

# Création de données simulées par classe
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

# Nombre d'échantillons par classe (environ équilibré)
n_per_class = n_samples // 3

df_normal = generate_class_data("Normal", n_per_class)
df_inner = generate_class_data("Inner_Fault", n_per_class)
df_outer = generate_class_data("Outer_Fault", n_samples - 2*n_per_class)

# Concaténation
df = pd.concat([df_normal, df_inner, df_outer], ignore_index=True)

# Sauvegarde CSV
df.to_csv("vibrations_pieces.csv", index=False)

print("Dataset simulé créé et sauvegardé sous 'vibrations_pieces.csv' avec", len(df), "échantillons.")
