import json
import random
from pathlib import Path

# Configurações
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "contratos_reais_2000.json"
NUM_EXAMPLES = 1000  # 1000 rótulo 1, 1000 rótulo 0

# Dados para variações
nomes = [
    "Ana Maria Silva", "João Pedro Santos", "Maria Oliveira Lima", "Carlos Eduardo Souza",
    "Fernanda Costa Almeida", "Pedro Henrique Lima", "Laura Mendes Ribeiro", "Rafael Almeida Santos",
    "Beatriz Cristina Ferreira", "Lucas Gabriel Pereira", "Camila Souza Ribeiro", "Thiago Augusto Martins",
    "Juliana Castro Mendes", "Gabriel Rocha Silva", "Mariana Torres Almeida", "Eduardo Lima Santos",
    "Carla Mendes Ferreira", "Felipe Augusto Costa", "Sofia Ribeiro Lima", "André Oliveira Santos"
]
administradoras = [
    "Itaú Consórcios", "Bradesco Consórcios", "Caixa Consórcios", "Santander Consórcios",
    "Porto Seguro Consórcios", "Rodobens Consórcios", "Banco do Brasil Consórcios", "HSBC Consórcios"
]
fundos = [
    "Fundo de Investimento XYZ", "Fundo ABC", "Fundo DEF", "Fundo GHI", "Fundo JKL",
    "Fundo MNO", "Fundo PQR", "Fundo STU"
]
cidades = [
    "São Paulo, SP", "Rio de Janeiro, RJ", "Belo Horizonte, MG", "Curitiba, PR",
    "Porto Alegre, RS", "Recife, PE", "Salvador, BA", "Brasília, DF"
]
tipos_bem = ["móvel", "imóvel", "veículo"]
tipos_locacao = ["residencial", "comercial"]
servicos = [
    "consultoria financeira", "marketing digital", "desenvolvimento de software",
    "limpeza predial", "segurança patrimonial", "auditoria contábil", "treinamento corporativo",
    "manutenção de equipamentos"
]
veiculos = [
    "Fiat Argo 2023", "Volkswagen Gol 2022", "Toyota Corolla 2024", "Honda Civic 2023",
    "Chevrolet Onix 2022", "Hyundai HB20 2023", "Ford Ka 2022", "Renault Kwid 2024"
]
imoveis = [
    "apartamento de 70m²", "casa de 120m²", "sala comercial de 50m²", "loja de 100m²",
    "terreno de 300m²", "cobertura de 150m²"
]
datas = [
    "10/03/2025", "15/04/2025", "20/05/2025", "01/06/2025", "30/06/2025",
    "05/07/2025", "12/08/2025", "25/09/2025"
]

# Função para gerar contrato de cessão (rótulo 1)
def gerar_cessao():
    cedente = random.choice(nomes)
    cessionario = random.choice(fundos)
    cota = random.randint(100, 9999)
    grupo = random.randint(10000, 99999)
    tarifa = random.randint(500, 2000)
    saldo = random.randint(40, 80)
    admin = random.choice(administradoras)
    bem = random.choice(tipos_bem)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    template = random.choice([
        f"Instrumento Particular de Cessão de Cota de Consórcio de Bem {bem.capitalize()}. A cedente, {cedente}, transfere ao cessionário, {cessionario}, todos os direitos e obrigações referentes à cota {cota} do grupo {grupo}, administrada pela {admin}. A cessão, formalizada em {data} na cidade de {cidade}, inclui tarifa de R${tarifa},00, e o cessionário assume {saldo}% do saldo devedor, conforme cláusula 4 do contrato de adesão.",
        f"Aditamento ao Contrato de Consórcio de Bem {bem.capitalize()}. A cedente, {cedente}, cede ao cessionário, {cessionario}, a cota {cota} do grupo {grupo}, com tarifa de cessão de R${tarifa},00. O cessionário assume todas as obrigações financeiras, incluindo {saldo}% das parcelas restantes, conforme contrato firmado com a {admin} em {data}. O foro de {cidade} é competente para dirimir dúvidas.",
        f"Contrato de Transferência de Cota de Consórcio. Por este instrumento, a cedente, {cedente}, transfere ao cessionário, {cessionario}, a cota {cota} do grupo {grupo}, administrada pela {admin}, com tarifa de R${tarifa},00. O cessionário assume {saldo}% do saldo devedor, conforme estipulado em {data}. A cessão é regida pelas normas do contrato original e formalizada em {cidade}.",
        f"Termo de Cessão de Direitos de Cota de Consórcio. A cedente, {cedente}, transfere ao cessionário, {cessionario}, a cota {cota} do grupo {grupo}, com tarifa de cessão de R${tarifa},00, conforme contrato de adesão da {admin}. O cessionário assume {saldo}% das parcelas, em acordo firmado em {data} na cidade de {cidade}, sob as condições do contrato original."
    ])
    return {"texto": template, "rotulo": 1}

# Função para gerar outros contratos (rótulo 0)
def gerar_outro():
    tipo_contrato = random.choice(["locacao", "servico", "emprestimo", "compra_venda", "consorcio"])
    cidade = random.choice(cidades)
    data = random.choice(datas)

    if tipo_contrato == "locacao":
        locador = random.choice(nomes)
        locatario = random.choice(nomes)
        tipo_loc = random.choice(tipos_locacao)
        imovel = random.choice(imoveis)
        valor = random.randint(1500, 12000)
        prazo = random.choice([12, 24, 36, 48])
        template = f"Contrato de Locação {tipo_loc.capitalize()}. O locador, {locador}, aluga ao locatário, {locatario}, o {imovel} situado na Rua das Acácias, 456, {cidade}, pelo valor mensal de R${valor},00. O contrato, firmado em {data}, tem prazo de {prazo} meses, com renovação automática, conforme cláusula 5."

    elif tipo_contrato == "servico":
        contratada = f"Empresa {random.choice(['ABC', 'DEF', 'GHI', 'JKL'])} Ltda"
        contratante = random.choice(nomes)
        servico = random.choice(servicos)
        valor = random.randint(2000, 10000)
        duracao = random.choice([6, 12, 18, 24])
        template = f"Contrato de Prestação de Serviços. A contratada, {contratada}, compromete-se a prestar serviços de {servico} à contratante, {contratante}, pelo valor de R${valor},00, com duração de {duracao} meses. O contrato, assinado em {data} em {cidade}, inclui cronograma detalhado anexo."

    elif tipo_contrato == "emprestimo":
        credor = f"Banco {random.choice(['XYZ', 'DEF', 'GHI', 'JKL'])}"
        devedor = random.choice(nomes)
        valor = random.randint(10000, 100000)
        juros = round(random.uniform(1, 3.5), 1)
        parcelas = random.choice([12, 24, 36, 48])
        template = f"Contrato de Mútuo. O mutuante, {credor}, concede ao mutuário, {devedor}, o empréstimo de R${valor},00, com juros de {juros}% ao mês, a serem pagos em {parcelas} parcelas mensais. O contrato, firmado em {data} em {cidade}, inclui cláusula de garantia hipotecária."

    elif tipo_contrato == "compra_venda":
        vendedor = random.choice(nomes)
        comprador = random.choice(nomes)
        objeto = random.choice(veiculos + imoveis)
        valor = random.randint(40000, 500000)
        pagamento = random.choice(["parcela única", f"{random.randint(6, 24)} parcelas"])
        template = f"Contrato de Compra e Venda. O vendedor, {vendedor}, transfere ao comprador, {comprador}, o {objeto}, pelo valor de R${valor},00, pago em {pagamento}. O contrato, assinado em {data} em {cidade}, inclui recibo de entrega."

    else:  # Consórcio não-cessão
        consorciado = random.choice(nomes)
        cota = random.randint(100, 9999)
        grupo = random.randint(10000, 99999)
        admin = random.choice(administradoras)
        parcelas = random.choice([60, 120, 180, 240])
        valor = random.randint(1000, 4000)
        bem = random.choice(tipos_bem)
        template = f"Contrato de Participação em Grupo de Consórcio de Bem {bem.capitalize()}. O consorciado, {consorciado}, adere ao grupo {grupo}, cota {cota}, administrado pela {admin}. O contrato, firmado em {data} em {cidade}, prevê {parcelas} parcelas de R${valor},00, sem cessão de direitos a terceiros."

    return {"texto": template, "rotulo": 0}

# Gerar dataset
dataset = []
for _ in range(1000):  # 1000 exemplos de rótulo 1
    dataset.append(gerar_cessao())
for _ in range(1000):  # 1000 exemplos de rótulo 0
    dataset.append(gerar_outro())

# Embaralhar dataset
random.shuffle(dataset)

# Salvar em JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset com {NUM_EXAMPLES} registros salvo em: {OUTPUT_PATH}")