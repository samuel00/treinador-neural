import pytest
import torch
from src.bert.treinador import ContractDataset, preprocess_text
from transformers import BertTokenizer


@pytest.mark.parametrize("entrada,esperado", [
    ("Texto Normal", "texto normal"),
    ("  Com espaço ", "com espaço"),
])
def test_preprocess_text_simples(entrada, esperado):
    assert preprocess_text(entrada) == esperado


def test_contract_dataset_formato():
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    texts = ["Contrato A", "Contrato B"]
    labels = [1, 0]
    ds = ContractDataset(texts, labels, tokenizer, max_len=64)
    item = ds[0]
    assert isinstance(item, dict)
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape[0] == 64
    assert item["labels"].item() in [0, 1]


def test_contract_dataset_len():
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    texts = ["Texto 1", "Texto 2", "Texto 3"]
    labels = [0, 1, 1]
    ds = ContractDataset(texts, labels, tokenizer)
    assert len(ds) == 3


def test_contract_dataset_input_variacao():
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    textos = ["Um contrato com muitas palavras " * 10]
    labels = [1]
    ds = ContractDataset(textos, labels, tokenizer, max_len=128)
    item = ds[0]
    assert item["input_ids"].shape[0] == 128
