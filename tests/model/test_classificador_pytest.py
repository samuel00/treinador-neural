from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
import torch

from bert.classificador import preprocess_text, dividir_em_trechos, gerar_embedding, MODEL_PATH, carregar_modelo_bert


@pytest.mark.parametrize("texto,esperado", [
    ("Texto Exemplo!!! 123", "texto exemplo"),
    ("  Com espaços   extras. ", "com espaços   extras"),
    ("Somente Letras", "somente letras")
])
def test_preprocess_text(texto, esperado):
    assert preprocess_text(texto) == esperado
    

def test_dividir_em_trechos_exato():
    texto = "palavra " * 300
    trechos = dividir_em_trechos(texto.strip(), tamanho=100)
    assert len(trechos) == 3
    assert all(len(t.split()) <= 100 for t in trechos)
    

def test_gerar_embedding_mock(monkeypatch):
    class MockBertOutput:
        def __init__(self):
            self.last_hidden_state = torch.rand(1, 10, 768)

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones((1, 10))
    }

    mock_model = MagicMock()
    mock_model.return_value = MockBertOutput()
    mock_model.config.hidden_size = 768

    monkeypatch.setattr("src.bert.classificador.bert_tokenizer", mock_tokenizer)
    monkeypatch.setattr("src.bert.classificador.bert_model", mock_model)

    texto = "Teste do embedding gerado com mock"
    embedding = gerar_embedding(texto)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)
    assert not np.isnan(embedding).any()
    

def test_carregar_modelo_bert_existe(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)

    mock_model = {"modelo": "teste"}

    with patch("builtins.open", mock_open(read_data=b"123")), \
            patch("pickle.load", return_value=mock_model):
        resultado = carregar_modelo_bert()
        assert resultado == mock_model
        

def test_carregar_modelo_bert_nao_existe(monkeypatch):
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            carregar_modelo_bert()