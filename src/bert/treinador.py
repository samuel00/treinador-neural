import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import re

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "bert_finetuned"
BERT_NAME = "neuralmind/bert-base-portuguese-cased"

# Função para pré-processar texto (menos agressiva)
def preprocess_text(text):
    text = text.lower()  # Apenas normalização para minúsculas
    return text.strip()

# Dataset personalizado
class ContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def treinar_modelo_bert(json_path):
    # Carregar dados
    with open(json_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    textos = [preprocess_text(d["texto"].strip()) for d in dados]
    rotulos = [d["rotulo"] for d in dados]

    # Verificar balanceamento
    print(f"Total de exemplos: {len(dados)}")
    print(f"Rótulo 0: {sum(r == 0 for r in rotulos)}")
    print(f"Rótulo 1: {sum(r == 1 for r in rotulos)}")

    # Divisão treino/validação/teste
    X_train, X_temp, y_train, y_temp = train_test_split(
        textos, rotulos, test_size=0.3, stratify=rotulos, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Verificar tamanhos dos conjuntos
    print(f"Treino: {len(X_train)} exemplos")
    print(f"Validação: {len(X_val)} exemplos")
    print(f"Teste: {len(X_test)} exemplos")
    print(f"Rótulo 0 no teste: {sum(y == 0 for y in y_test)}")
    print(f"Rótulo 1 no teste: {sum(y == 1 for y in y_test)}")

    # Carregar tokenizer e modelo
    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_NAME, num_labels=2)

    # Criar datasets
    train_dataset = ContractDataset(X_train, y_train, tokenizer)
    val_dataset = ContractDataset(X_val, y_val, tokenizer)
    test_dataset = ContractDataset(X_test, y_test, tokenizer)

    # Lidar com desbalanceamento
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Pesos de classe: {class_weights}")

    # Modificar função de perda
    def compute_loss(model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    model.compute_loss = compute_loss

    # Configurar treinamento
    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        num_train_epochs=10,  # Aumentado para aprendizado mais robusto
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.02,  # Aumentado para evitar overfitting
        learning_rate=2e-5,  # Reduzido para convergência mais estável
        logging_dir=str(MODEL_PATH / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Função para métricas
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted')
        }

    # Treinar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Avaliar no conjunto de teste
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = y_test

    # Exibir métricas
    print("\n--- Resultados no Conjunto de Teste ---")
    print("F1-score:", f1_score(y_true, y_pred, average='weighted'))
    print("Precisão:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("Matriz de confusão:\n", confusion_matrix(y_true, y_pred))

    # Salvar modelo
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    json_path = BASE_DIR / "data" / "contratos_ideal_1000.json"
    treinar_modelo_bert(json_path)