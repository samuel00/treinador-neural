import unittest
from unittest.mock import patch, Mock, mock_open, MagicMock
from pathlib import Path
import numpy as np
import pytest
import torch

from src.bert.classificador import preprocess_text, dividir_em_trechos, gerar_embedding, carregar_modelo_bert

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "src" / "model" / "modelo_bert.pkl"


class TestClassificador(unittest.TestCase):

    def test_preprocess_text(self):
        # Testar normalização e limpeza
        texto = "Contrato de LOCAÇÃO 123! @#$"
        resultado = preprocess_text(texto)
        self.assertEqual(resultado, "contrato de locação")

        # Testar texto vazio
        self.assertEqual(preprocess_text(""), "")

        # Testar texto com apenas números
        self.assertEqual(preprocess_text("123"), "")

        # Testar texto com pontuação
        self.assertEqual(preprocess_text("teste,.;"), "teste")

    def test_dividir_em_trechos(self):
        # Testar divisão com tamanho padrão
        texto = "Contrato de locação residencial com valor de mil reais por mês"
        trechos = dividir_em_trechos(texto, tamanho=3)
        self.assertEqual(len(trechos), 4)
        self.assertEqual(trechos[0], "Contrato de locação")
        self.assertEqual(trechos[-1], "por mês")

        # Testar texto curto
        texto = "Contrato"
        trechos = dividir_em_trechos(texto)
        self.assertEqual(trechos, ["Contrato"])

        # Testar texto vazio
        self.assertEqual(dividir_em_trechos(""), [])

    @patch('src.bert.classificador.BertTokenizer')
    @patch('src.bert.classificador.BertModel')
    def test_gerar_embedding(self, mock_bert, mock_tokenizer):
        # Configurar mock do tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[101, 2000, 102]]),  # [CLS], "texto", [SEP]
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        # Configurar mock do modelo
        mock_bert_instance = Mock()
        mock_bert.from_pretrained.return_value = mock_bert_instance
        mock_output = Mock()
        mock_output.last_hidden_state = torch.zeros((1, 3, 768))  # Shape: (batch_size, seq_len, hidden_size)
        mock_bert_instance.return_value = mock_output
        mock_bert_instance.config.hidden_size = 768  # Configurar hidden_size para o fallback

        # Chamar a função
        embedding = gerar_embedding("texto de teste")

        # Depuração
        print(f"Embedding shape: {embedding.shape}, Embedding: {embedding[:5]}")

        # Verificar o resultado
        self.assertEqual(embedding.shape, (768,))  # Forma correta do embedding
        self.assertTrue(np.all(embedding == 0))  # Verifica se todos os valores são zero

        # Verificar chamadas aos mocks
        mock_tokenizer.from_pretrained.assert_called_once_with("neuralmind/bert-base-portuguese-cased")
        mock_bert.from_pretrained.assert_called_once_with("neuralmind/bert-base-portuguese-cased")

    @patch('src.bert.classificador.BertTokenizer')
    @patch('src.bert.classificador.BertModel')
    def test_gerar_embedding_empty_text(self, mock_bert, mock_tokenizer):
        # Configurar mocks para texto vazio
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value.return_value = {
            'input_ids': np.array([[101, 102]]),  # Apenas tokens especiais
            'attention_mask': np.array([[1, 1]])
        }
        mock_bert_instance = Mock()
        mock_bert.from_pretrained.return_value = mock_bert_instance
        mock_output = Mock()
        mock_output.last_hidden_state = np.zeros((1, 2, 768))
        mock_bert_instance.return_value = mock_output

        # Chamar a função
        embedding = gerar_embedding("")
        self.assertEqual(embedding.shape, (768,))

    @patch('builtins.open')
    @patch('pickle.load')
    def test_carregar_modelo_bert(self, mock_pickle_load, mock_open):
        # Testar carregamento bem-sucedido
        mock_pickle_load.return_value = Mock()
        modelo = carregar_modelo_bert()
        mock_open.assert_called_with(MODEL_PATH, "rb")
        self.assertIsNotNone(modelo)

        # Testar arquivo inexistente
        mock_open.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            carregar_modelo_bert()

    def test_gerar_embedding_error(self):
        # Testar entrada inválida
        with self.assertRaises(AttributeError):
            gerar_embedding(None)


if __name__ == '__main__':
    unittest.main()