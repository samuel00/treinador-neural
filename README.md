# treinador-neural


pip freeze > requirements.txt
pip install -r requirements.txt







# Exibir métricas
print("\n--- Resultados no Conjunto de Teste ---")
print("F1-score:", f1_score(y_true, y_pred, average='weighted'))
print("Precisão:", precision_score(y_true, y_pred, average='weighted'))
print("Recall:", recall_score(y_true, y_pred, average='weighted'))
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:")
print(f"  Verdadeiro Negativo (Classe 0 correta): {cm[0,0]}")
print(f"  Falso Positivo (Classe 0 errada): {cm[0,1]}")
print(f"  Falso Negativo (Classe 1 errada): {cm[1,0]}")
print(f"  Verdadeiro Positivo (Classe 1 correta): {cm[1,1]}")

# Salvar resultados em CSV
import pandas as pd


Analise Visual:

import pandas as pd
df = pd.read_csv("model/bert_finetuned/avaliacao_teste.csv")
print(df[df["true_label"] == 1][["texto_inicial", "prob_1", "pred_label"]])