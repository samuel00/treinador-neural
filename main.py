from pathlib import Path

from src.bert.treinador import treinar_modelo_bert

if __name__ == "__main__":
    caminho_json = Path(__file__).resolve().parent / "src" / "data" / "contratos_treinamento.json"
    print(f"Treinando modelo BERT com base: {caminho_json}")
    treinar_modelo_bert(str(caminho_json))
