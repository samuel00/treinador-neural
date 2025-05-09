import pickle
from pathlib import Path
import re
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Caminhos de modelo e tokenizer
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "modelo_bert.pkl"
BERT_NAME = "neuralmind/bert-base-portuguese-cased"
RB_MODE = "rb"

# Variáveis globais para armazenar tokenizer e modelo (inicializadas como None)
bert_tokenizer = None
bert_model = None

# Função para inicializar tokenizer e modelo sob demanda
def initialize_bert():
    global bert_tokenizer, bert_model
    if bert_tokenizer is None or bert_model is None:
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
        bert_model = BertModel.from_pretrained(BERT_NAME)
    return bert_tokenizer, bert_model

# Função para pré-processar texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove números
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    return text.strip()

# Função para dividir em trechos
def dividir_em_trechos(texto, tamanho=100):
    palavras = texto.split()
    return [" ".join(palavras[i:i + tamanho]) for i in range(0, len(palavras), tamanho)]

# Função para gerar embeddings com BERT
@torch.no_grad()
def gerar_embedding(texto, max_length=512):
    tokenizer, model = initialize_bert()
    texto = preprocess_text(texto)
    trechos = dividir_em_trechos(texto)
    embeddings = []

    for trecho in trechos:
        inputs = tokenizer(
            trecho,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=max_length
        )
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # [CLS] token
        # Verificar se é um tensor e converter para numpy
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        embeddings.append(embedding)

    # Combinar embeddings (média)
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768)

# Carregar modelo salvo
def carregar_modelo_bert():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo BERT ainda não foi treinado. Execute treinar_modelo_bert primeiro.")
    with open(MODEL_PATH, RB_MODE) as f:
        return pickle.load(f)