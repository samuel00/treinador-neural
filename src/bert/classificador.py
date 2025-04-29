import pickle
from pathlib import Path

import torch
from transformers import BertTokenizer, BertModel

# Caminhos de modelo e tokenizer
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "modelo_bert.pkl"
BERT_NAME = "neuralmind/bert-base-portuguese-cased"
LANGUAGE = "pt"
RB_MODE = "rb"

# Inicializa tokenizer e modelo BERT
bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
bert_model = BertModel.from_pretrained(BERT_NAME)

# Função para dividir em trechos
def dividir_em_trechos(texto, tamanho=100):
    palavras = texto.split()
    return [" ".join(palavras[i:i + tamanho]) for i in range(0, len(palavras), tamanho)]

# Função para gerar embeddings com BERT
@torch.no_grad()
def gerar_embedding(texto):
    inputs = bert_tokenizer(texto, return_tensors=LANGUAGE, truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token
    return embedding

# Carregar modelo salvo
def carregar_modelo_bert():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo BERT ainda não foi treinado. Execute treinar_modelo_bert primeiro.")
    with open(MODEL_PATH, RB_MODE) as f:
        return pickle.load(f)
