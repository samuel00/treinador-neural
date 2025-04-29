# Treinamento inicial e salvamento
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.bert.classificador import gerar_embedding
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "modelo_bert.pkl"

# Treinamento inicial e salvamento
def treinar_modelo_bert(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    textos = [d["texto"].strip() for d in dados]
    rotulos = [d["rotulo"] for d in dados]

    embeddings = []
    for texto in tqdm(textos, desc="Gerando embeddings BERT"):
        emb = gerar_embedding(texto)
        embeddings.append(emb)

    X = np.array(embeddings)
    y = np.array(rotulos)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\n--- Relatório de Classificação (validação) ---")
    print(classification_report(y_val, y_pred, digits=4))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Modelo salvo em: {MODEL_PATH}")