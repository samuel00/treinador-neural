import json
import random
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "bert_finetuned"
BERT_NAME = "neuralmind/bert-base-portuguese-cased"
ONNX_MODEL_PATH = MODEL_PATH / "bert_finetuned.onnx"

# Fixar sementes para reprodutibilidade
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# Função para pré-processar texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Normalizar espaços
    return text.strip()


# Dataset personalizado com divisão de trechos
class ContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256, chunk_size=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.chunks, self.chunk_labels, self.doc_indices = self._create_chunks()

    def _create_chunks(self):
        chunks = []
        chunk_labels = []
        doc_indices = []
        for doc_idx, (text, label) in enumerate(zip(self.texts, self.labels)):
            words = text.split()
            for i in range(0, len(words), self.chunk_size):
                chunk = ' '.join(words[i:i + self.chunk_size])
                chunks.append(chunk)
                chunk_labels.append(label)
                doc_indices.append(doc_idx)  # Rastrear índice do documento original
        return chunks, chunk_labels, doc_indices

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        text = self.chunks[idx]
        label = self.chunk_labels[idx]
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
            'labels': torch.tensor(label, dtype=torch.long),
            'doc_index': self.doc_indices[idx]  # Incluir índice do documento
        }


# Função para agregar previsões por documento
def aggregate_predictions(predictions, doc_indices, num_docs):
    # Agregar probabilidades por documento (média das probabilidades dos trechos)
    doc_probs = np.zeros((num_docs, 2))  # Para 2 classes
    doc_counts = np.zeros(num_docs, dtype=int)

    for prob, doc_idx in zip(predictions, doc_indices):
        doc_probs[doc_idx] += prob
        doc_counts[doc_idx] += 1

    # Calcular média das probabilidades
    doc_probs = np.array([prob / count if count > 0 else prob for prob, count in zip(doc_probs, doc_counts)])

    # Prever classe com base na maior probabilidade
    doc_preds = np.argmax(doc_probs, axis=1)
    return doc_preds, doc_probs


# Função para exportar o modelo para ONNX
def export_to_onnx(model, tokenizer, onnx_output_path, max_len=256):
    try:
        print("Iniciando exportação para ONNX...")
        onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        model = model.to("cpu")
        dummy_text = "Aditamento ao contrato de participação em grupo de consórcio para cessão de direitos e obrigações."
        inputs = tokenizer(
            dummy_text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        print("Exportando modelo para ONNX com opset_version=14...")
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=False,  # Desativar para consistência
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"}
            }
        )
        if onnx_output_path.exists():
            print(f"Modelo exportado com sucesso para: {onnx_output_path}")
        else:
            print(f"Erro: Arquivo ONNX não foi criado em: {onnx_output_path}")
    except Exception as e:
        print(f"Erro ao exportar modelo para ONNX: {str(e)}")
        raise


# Função para validar o modelo ONNX
def validate_onnx_model(model_pytorch, onnx_path, tokenizer, test_texts, max_len=256):
    try:
        print("Validando modelo ONNX...")
        session = ort.InferenceSession(str(onnx_path))
        for text in test_texts[:5]:
            inputs = tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].numpy()
            attention_mask = inputs['attention_mask'].numpy()
            ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            ort_outs = session.run(None, ort_inputs)
            onnx_pred = np.argmax(ort_outs[0], axis=-1)
            model_pytorch.eval()
            with torch.no_grad():
                outputs = model_pytorch(input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
                pytorch_pred = torch.argmax(outputs.logits, dim=-1).numpy()
            print(f"Texto: {text[:50]}...")
            print(f"ONNX Predição: {onnx_pred}, PyTorch Predição: {pytorch_pred}")
    except Exception as e:
        print(f"Erro ao validar modelo ONNX: {str(e)}")


def treinar_modelo_bert_onnx(json_path):
    # Carregar dados
    try:
        print("Carregando dados...")
        with open(json_path, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar JSON: {str(e)}")
        raise

    textos = [preprocess_text(d["texto"].strip()) for d in dados]
    rotulos = [d["rotulo"] for d in dados]

    # Verificar balanceamento
    print(f"Total de exemplos: {len(dados)}")
    print(f"Rótulo 0 (outros documentos): {sum(r == 0 for r in rotulos)}")
    print(f"Rótulo 1 (cessão de consórcio): {sum(r == 1 for r in rotulos)}")

    # Divisão treino/validação/teste
    print("Dividindo dados...")
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
    print("Carregando tokenizer e modelo...")
    tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_NAME, num_labels=2)
    model = model.to("cpu")  # Forçar CPU para consistência

    # Criar datasets
    print("Criando datasets...")
    train_dataset = ContractDataset(X_train, y_train, tokenizer, max_len=256, chunk_size=200)
    val_dataset = ContractDataset(X_val, y_val, tokenizer, max_len=256, chunk_size=200)
    test_dataset = ContractDataset(X_test, y_test, tokenizer, max_len=256, chunk_size=200)

    # Lidar com desbalanceamento
    print("Calculando pesos de classe...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cpu')
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
    print("Configurando argumentos de treinamento...")
    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=str(MODEL_PATH / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
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
    print("Iniciando treinamento...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Avaliar no conjunto de teste
    print("Avaliando no conjunto de teste...")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    doc_indices = [item['doc_index'] for item in test_dataset]
    num_docs = len(y_test)

    # Agregar previsões por documento
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred, doc_probs = aggregate_predictions(probs, doc_indices, num_docs)
    y_true = y_test

    # Exibir métricas
    print("\n--- Resultados no Conjunto de Teste ---")
    print("F1-score:", f1_score(y_true, y_pred, average='weighted'))
    print("Precisão:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de Confusão:")
    print(f"  Verdadeiro Negativo (Classe 0 correta): {cm[0, 0]}")
    print(f"  Falso Positivo (Classe 0 errada): {cm[0, 1]}")
    print(f"  Falso Negativo (Classe 1 errada): {cm[1, 0]}")
    print(f"  Verdadeiro Positivo (Classe 1 correta): {cm[1, 1]}")

    df_resultados = pd.DataFrame({
        "doc_id": range(len(y_true)),
        "texto_inicial": [text[:50] + "..." for text in X_test],
        "prob_0": doc_probs[:, 0],
        "prob_1": doc_probs[:, 1],
        "true_label": y_true,
        "pred_label": y_pred,
        "correto": [1 if true == pred else 0 for true, pred in zip(y_true, y_pred)]
    })
    df_resultados.to_csv(MODEL_PATH / "avaliacao_teste.csv", index=False, encoding="utf-8")
    print(f"Resultados salvos em: {MODEL_PATH / 'avaliacao_teste.csv'}")

    # Salvar modelo e tokenizer
    print("Salvando modelo e tokenizer...")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH, legacy_format=False)
    print(f"Modelo salvo em: {MODEL_PATH}")

    # Exportar para ONNX
    print("Iniciando exportação para ONNX...")
    export_to_onnx(model, tokenizer, ONNX_MODEL_PATH)

    # Validar modelo ONNX
    print("Validando exportação ONNX...")
    validate_onnx_model(model, ONNX_MODEL_PATH, tokenizer, X_test)