# Étape 1 : base Python
FROM python:3.9-slim

# Étape 2 : définir le répertoire de travail
WORKDIR /app

# Étape 3 : copier tous les fichiers dans le conteneur
COPY . .

# Étape 4 : installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : générer les données + entraîner le modèle
RUN python train.py

# Étape 6 : lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
