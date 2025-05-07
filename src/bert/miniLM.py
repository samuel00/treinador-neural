

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