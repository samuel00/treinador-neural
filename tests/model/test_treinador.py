import unittest
from unittest.mock import patch, Mock
from pathlib import Path
import sys
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Adicionar diretório raiz ao sys.path
from sklearn.utils import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from src.bert.treinador import preprocess_text, ContractDataset, treinar_modelo_bert

TEST_JSON_PATH = BASE_DIR / "data" / "contratos_teste.json"

class TestTreinador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Criar arquivo JSON de teste com 20 exemplos
        cls.test_data = [
            {"texto": "Contrato de locação residencial. Locador: João Silva.", "rotulo": 0},
            {"texto": "Instrumento de cessão de cota. Cedente: Ana Souza.", "rotulo": 1},
            {"texto": "Contrato de prestação de serviços. Contratada: Empresa ABC.", "rotulo": 0},
            {"texto": "Aditamento de consórcio. Cedente: Pedro Santos.", "rotulo": 1},
            {"texto": "Contrato de mútuo. Mutuante: Banco XYZ.", "rotulo": 0},
            {"texto": "Termo de cessão de cota. Cedente: Lucas Pereira.", "rotulo": 1},
            {"texto": "Contrato de compra e venda. Vendedor: Mariana Costa.", "rotulo": 0},
            {"texto": "Contrato de consórcio. Cedente: Beatriz Ferreira.", "rotulo": 1},
            {"texto": "Contrato de locação comercial. Locador: Empresa DEF.", "rotulo": 0},
            {"texto": "Cessão de cota de consórcio. Cedente: Camila Ribeiro.", "rotulo": 1},
            {"texto": "Contrato de locação de veículo. Locador: José Lima.", "rotulo": 0},
            {"texto": "Termo de cessão de cota. Cedente: Fernanda Almeida.", "rotulo": 1},
            {"texto": "Contrato de prestação de serviços técnicos. Empresa: Tech XYZ.", "rotulo": 0},
            {"texto": "Aditamento de contrato de consórcio. Cedente: Roberto Dias.", "rotulo": 1},
            {"texto": "Contrato de empréstimo. Credor: Banco ABC.", "rotulo": 0},
            {"texto": "Cessão de cota de consórcio. Cedente: Mariana Lopes.", "rotulo": 1},
            {"texto": "Contrato de venda de imóvel. Vendedor: Carlos Eduardo.", "rotulo": 0},
            {"texto": "Instrumento de cessão de cota. Cedente: Sofia Mendes.", "rotulo": 1},
            {"texto": "Contrato de locação de equipamento. Locador: Empresa GHI.", "rotulo": 0},
            {"texto": "Contrato de consórcio com cessão. Cedente: André Silva.", "rotulo": 1}
        ]
        with open(TEST_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(cls.test_data, f, ensure_ascii=False, indent=2)

    def test_preprocess_text(self):
        texto = "Contrato de LOCAÇÃO Residencial! 123"
        resultado = preprocess_text(texto)
        self.assertEqual(resultado, "contrato de locação residencial! 123")
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text("  Texto  "), "texto")

    def test_contract_dataset(self):
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        texts = ["Contrato de locação", "Cessão de cota"]
        labels = [0, 1]
        dataset = ContractDataset(texts, labels, tokenizer, max_len=128)
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        self.assertEqual(item['labels'].item(), 0)
        dataset = ContractDataset([""], [0], tokenizer, max_len=128)
        item = dataset[0]
        self.assertEqual(item['labels'].item(), 0)

    @patch('src.bert.treinador.BertTokenizer.from_pretrained')
    @patch('src.bert.treinador.BertForSequenceClassification.from_pretrained')
    @patch('src.bert.treinador.Trainer')
    def test_treinar_modelo_bert(self, mock_trainer, mock_model, mock_tokenizer):
        # Configurar mocks
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        # Configurar mock do Trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        # Simular o resultado de predict com logits
        mock_prediction_output = Mock()
        mock_prediction_output.predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])  # Logits para 3 exemplos
        mock_trainer_instance.predict.return_value = mock_prediction_output

        # Simular o treinamento (evitar execução real)
        mock_trainer_instance.train.return_value = None

        # Executar a função que estamos testando
        treinar_modelo_bert(TEST_JSON_PATH)

        # Verificar chamadas aos mocks
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.predict.assert_called_once()

        # Carregar os dados e verificar a divisão
        with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
        textos = [preprocess_text(d["texto"]) for d in dados]
        rotulos = [d["rotulo"] for d in dados]

        # Verificar total de exemplos e balanceamento
        self.assertEqual(len(dados), 20)
        self.assertEqual(sum(r == 0 for r in rotulos), 10)
        self.assertEqual(sum(r == 1 for r in rotulos), 10)

        # Simular a primeira divisão
        X_train, X_temp, y_train, y_temp = train_test_split(
            textos, rotulos, test_size=0.3, stratify=rotulos, random_state=42
        )
        # Simular a segunda divisão
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # Verificar tamanhos dos conjuntos
        self.assertEqual(len(X_train), 14)  # 70% de 20
        self.assertEqual(len(X_temp), 6)    # 30% de 20
        self.assertEqual(len(X_val), 3)     # 15% de 20
        self.assertEqual(len(X_test), 3)    # 15% de 20

        # Verificar balanceamento em X_temp
        self.assertEqual(sum(y == 0 for y in y_temp), 3)
        self.assertEqual(sum(y == 1 for y in y_temp), 3)

    def test_class_weights(self):
        y_train = [0, 0, 0, 1, 1]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        self.assertAlmostEqual(class_weights[0], 0.8333333333333334, places=7)  # Ajustado para o valor correto
        self.assertAlmostEqual(class_weights[1], 1.25, places=7)  # Ajustado para o valor correto

    def test_preprocess_text_error(self):
        with self.assertRaises(AttributeError):
            preprocess_text(None)

    @classmethod
    def tearDownClass(cls):
        if TEST_JSON_PATH.exists():
            TEST_JSON_PATH.unlink()

if __name__ == '__main__':
    unittest.main()