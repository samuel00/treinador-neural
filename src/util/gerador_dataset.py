import json
import random
from pathlib import Path
import faker
import re

# Configurações
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "contratos_cessao_5000.json"
NUM_EXAMPLES = 5000  # 2500 rótulo 1 (cessão de consórcio), 2500 rótulo 0 (outros documentos)
faker.Faker.seed(42)
random.seed(42)

# Inicializar o Faker
fake = faker.Faker('pt_BR')

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
datas = [
    "10/03/2025", "15/04/2025", "20/05/2025", "01/06/2025", "30/06/2025",
    "05/07/2025", "12/08/2025", "25/09/2025"
]
veiculos = [
    "Fiat Argo 2023", "Volkswagen Gol 2022", "Toyota Corolla 2024", "Honda Civic 2023",
    "Chevrolet Onix 2022", "Hyundai HB20 2023", "Ford Ka 2022", "Renault Kwid 2024"
]
imoveis = [
    "apartamento de 70m²", "casa de 120m²", "sala comercial de 50m²", "loja de 100m²",
    "terreno de 300m²", "cobertura de 150m²"
]

# Texto base para cessão de consórcio (rótulo 1, limpo de artefatos)
CESSAO_CONSORCIO_TEXT = """Aditamento ao Contrato de Participação em Grupo de Consórcio de Bem Móvel/Imóvel - Cessão de Direitos e Obrigações  
1. Administradora/Credora Fiduciária {admin} com sede em {endereco_admin}, inscrita no CNPJ sob o nº {cnpj_admin}.  
2. Cedente  
2.1 Nome: {cedente}  
2.2 CPF/CNPJ: {cpf_cedente}  
3. Cessionário  
3.1 Nome: {cessionario}  
3.2 CPF/CNPJ: {cpf_cessionario}  
3.3 Endereço/Sede: {endereco_cessionario}  
3.4 Dados da Conta Corrente: Agência {agencia} Conta {conta} - DAC {dac}  
3.5. Telefones do cessionário: {telefone}  
4. Dados do Contrato  
4.1 Grupo: {grupo}  
4.2 Cota: {cota}  
4.3 Percentual Pago: {percentual_pago}%  
4.4 Percentual a Vencer: {percentual_vencer}%  
5. Modo de Pagamento  
6. Tarifa de Aditamento Contratual  
5.1 (X) documento de cobrança R${tarifa},00  
5.2 ( ) débito na conta corrente indicada no subitem  
A empresa indicada no item 1, doravante designada Administradora, e as pessoas qualificadas nos itens 2 e 3, designadas respectivamente, Cedente e Cessionário, aditam o Contrato de Participação em Grupo de Consórcio de Bem Móvel/Imóvel (“Contrato de Adesão”) indicado no item 4, de acordo com as cláusulas que seguem.  
7. Objeto - O Cedente, autorizado pela Administradora, transfere ao Cessionário todos direitos e obrigações previstos no Contrato de Adesão que regula o Grupo/Cota de consórcio apontados no item 4.  
7.1 O CESSIONÁRIO DECLARA QUE RECEBEU CÓPIA DO CONTRATO DE ADESÃO, O LEU PREVIAMENTE, NÃO TENDO QUALQUER DÚVIDA EM RELAÇÃO AO QUE ALI RESTOU DISPOSTO. CONCORDA COM TODAS AS CLÁUSULAS E CONDIÇÕES DO CONTRATO DE ADESÃO E, ATRAVÉS DESTE INSTRUMENTO, ASSUME TODAS AS OBRIGAÇÕES NELE PREVISTAS, EM ESPECIAL AS DE NATUREZA FINANCEIRA.  
8. Modo de Pagamento - O Cessionário fará os pagamentos na forma indicada no item 5.  
8.1. Sendo assinalado o subitem 5.2, o Cessionário pagará todos os valores devidos em decorrência do Contrato de Adesão mediante débito em sua conta corrente mantida junto ao {admin}, indicada no subitem 3.4, que deverá ter saldo disponível suficiente.  
8.2 A insuficiência de saldo na conta corrente apontada no subitem 3.4 configurará atraso no pagamento.  
8.3 Sendo indicado o subitem 5.1, o Cessionário fará todos os pagamentos por meio de documento de cobrança (carnê ou equivalente) a ser emitido pelo {admin} e enviado para o endereço indicado no subitem 3.3.  
8.4 O Cessionário está ciente de que, havendo atraso no pagamento de qualquer parcela, ficará impedido de concorrer aos sorteios e de ofertar lances, sem prejuízo das demais sanções previstas no Contrato de Adesão.  
9. Tolerância — A tolerância das partes quanto ao descumprimento de qualquer obrigação pela outra parte não significará renúncia ao direito de exigir o cumprimento da obrigação, nem perdão, nem alteração do que foi aqui ajustado.  
10. Tarifa - O Cessionário pagará a taxa indicada no item 6 e prevista no Contrato de Adesão, em razão desta cessão de direitos e obrigações.  
11. Foro - Fica eleito o Foro da Comarca do local da assinatura deste Aditamento, podendo a parte que promover a ação optar pelo Foro do domicílio do Cessionário.  
Local e Data: {cidade}, {data}.  
Assinaturas:  
Cessionário: ___________________________  
Cedente: ___________________________  
Administradora: ___________________________  
Testemunhas:  
1. Nome: {testemunha1} CPF: {cpf_testemunha1}  
2. Nome: {testemunha2} CPF: {cpf_testemunha2}"""

# Função para gerar contrato de cessão de consórcio (rótulo 1)
def gerar_cessao_consorcio():
    cedente = random.choice(nomes)
    cessionario = random.choice(nomes + fundos)  # Pessoa física ou fundo
    admin = random.choice(administradoras)
    cota = random.randint(100, 9999)
    grupo = random.randint(10000, 99999)
    tarifa = random.randint(500, 2000)  # Tarifa variada (legítima ou suspeita)
    percentual_pago = round(random.uniform(10, 70), 2)  # Percentual pago variado
    percentual_vencer = round(100 - percentual_pago, 2)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_admin = fake.address().replace('\n', ', ')
    endereco_cessionario = fake.address().replace('\n', ', ')
    cnpj_admin = fake.cnpj()
    cpf_cedente = fake.cpf()
    cpf_cessionario = fake.cpf() if cessionario in nomes else fake.cnpj()
    agencia = random.randint(1000, 9999)
    conta = random.randint(10000, 99999)
    dac = random.randint(0, 9)
    telefone = fake.phone_number()
    testemunha1 = random.choice(nomes)
    testemunha2 = random.choice(nomes)
    cpf_testemunha1 = fake.cpf()
    cpf_testemunha2 = fake.cpf()
    return {
        "texto": CESSAO_CONSORCIO_TEXT.format(
            admin=admin, endereco_admin=endereco_admin, cnpj_admin=cnpj_admin,
            cedente=cedente, cpf_cedente=cpf_cedente,
            cessionario=cessionario, cpf_cessionario=cpf_cessionario,
            endereco_cessionario=endereco_cessionario, agencia=agencia, conta=conta, dac=dac,
            telefone=telefone, grupo=grupo, cota=cota, percentual_pago=percentual_pago,
            percentual_vencer=percentual_vencer, tarifa=tarifa, cidade=cidade, data=data,
            testemunha1=testemunha1, cpf_testemunha1=cpf_testemunha1,
            testemunha2=testemunha2, cpf_testemunha2=cpf_testemunha2
        ),
        "rotulo": 1
    }

# Função para gerar contrato de financiamento (rótulo 0)
def generate_financing_contract():
    financiador = fake.company()
    financiado = random.choice(nomes)
    cnpj_financiador = fake.cnpj()
    cpf_financiado = fake.cpf()
    valor = round(random.uniform(50000, 200000), 2)
    juros = round(random.uniform(1.5, 5.0), 2)
    parcelas = random.randint(12, 60)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    objeto = random.choice(veiculos + imoveis)
    endereco_financiador = fake.address().replace('\n', ', ')
    endereco_financiado = fake.address().replace('\n', ', ')
    return f"""Contrato de Financiamento nº {random.randint(1000, 9999)}  
1. Partes  
1.1. Financiador: {financiador}, com sede em {endereco_financiador}, inscrito no CNPJ sob o nº {cnpj_financiador}.  
1.2. Financiado: {financiado}, CPF nº {cpf_financiado}, residente em {endereco_financiado}.  
2. Objeto do Contrato  
O Financiador concede ao Financiado um financiamento no valor de R${valor}, destinado à aquisição de um {objeto}, conforme especificado no Anexo I.  
3. Condições de Pagamento  
3.1. O valor financiado será pago em {parcelas} parcelas mensais, com juros de {juros}% ao ano.  
3.2. A primeira parcela vencerá em {data}.  
3.3. O pagamento será realizado via débito automático na conta corrente do Financiado, indicada no Anexo II.  
4. Garantias  
O Financiado oferece como garantia o bem financiado, que será alienado fiduciariamente ao Financiador até a quitação total do financiamento.  
5. Inadimplemento  
Em caso de atraso no pagamento, será cobrada multa de 2% e juros de mora de 1% ao mês sobre o valor da parcela em atraso.  
6. Foro  
Fica eleito o foro da comarca de {cidade} para dirimir quaisquer dúvidas ou litígios decorrentes deste contrato.  
Local e Data: {cidade}, {data}.  
Assinaturas:  
Financiador: ___________________________  
Financiado: ___________________________  
Testemunhas:  
1. Nome: {random.choice(nomes)} CPF: {fake.cpf()}  
2. Nome: {random.choice(nomes)} CPF: {fake.cpf()}"""

# Função para gerar contrato de trabalho (rótulo 0)
def generate_employment_contract():
    empregador = fake.company()
    empregado = random.choice(nomes)
    cnpj_empregador = fake.cnpj()
    cpf_empregado = fake.cpf()
    cargo = fake.job()
    salario = round(random.uniform(2000, 10000), 2)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_empregador = fake.address().replace('\n', ', ')
    endereco_empregado = fake.address().replace('\n', ', ')
    return f"""Contrato de Trabalho nº {random.randint(1000, 9999)}  
1. Partes  
1.1. Empregador: {empregador}, com sede em {endereco_empregador}, inscrito no CNPJ sob o nº {cnpj_empregador}.  
1.2. Empregado: {empregado}, CPF nº {cpf_empregado}, residente em {endereco_empregado}.  
2. Objeto do Contrato  
O Empregador contrata o Empregado para exercer a função de {cargo}, com jornada de trabalho de 44 horas semanais.  
3. Remuneração  
3.1. O Empregado receberá o salário mensal de R${salario}, pago até o 5º dia útil de cada mês, via depósito na conta indicada no Anexo I.  
3.2. O Empregado terá direito a benefícios como vale-transporte e plano de saúde, conforme política interna da empresa.  
4. Prazo do Contrato  
Este contrato é por prazo indeterminado, com início em {data}, podendo ser rescindido por qualquer das partes mediante aviso prévio de 30 dias.  
5. Obrigações do Empregado  
5.1. Cumprir as normas internas da empresa e desempenhar suas funções com diligência.  
5.2. Manter sigilo sobre informações confidenciais da empresa.  
6. Foro  
Fica eleito o foro da comarca de {cidade} para dirimir quaisquer dúvidas ou litígios decorrentes deste contrato.  
Local e Data: {cidade}, {data}.  
Assinaturas:  
Empregador: ___________________________  
Empregado: ___________________________  
Testemunhas:  
1. Nome: {random.choice(nomes)} CPF: {fake.cpf()}  
2. Nome: {random.choice(nomes)} CPF: {fake.cpf()}"""

# Função para gerar fatura (rótulo 0)
def generate_invoice():
    empresa = fake.company()
    cliente = random.choice(nomes)
    cnpj_empresa = fake.cnpj()
    cpf_cliente = fake.cpf()
    valor = round(random.uniform(100, 5000), 2)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_empresa = fake.address().replace('\n', ', ')
    return f"""Fatura nº {random.randint(1000, 9999)}  
Emitente: {empresa}, CNPJ: {cnpj_empresa}, Endereço: {endereco_empresa}  
Destinatário: {cliente}, CPF: {cpf_cliente}  
Descrição: Prestação de serviços / Venda de produtos  
Valor Total: R${valor}  
Data de Emissão: {data}  
Data de Vencimento: {data}  
Forma de Pagamento: Boleto bancário  
Observações: Pagar até a data de vencimento para evitar multas.  
Local: {cidade}"""

# Função para gerar relatório (rótulo 0)
def generate_report():
    empresa = fake.company()
    autor = random.choice(nomes)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    return f"""Relatório Mensal - {empresa}  
Autor: {autor}  
Data: {data}  
Local: {cidade}  
Resumo: Este relatório apresenta os resultados operacionais do mês, incluindo vendas, despesas e metas alcançadas.  
1. Vendas: A empresa registrou um aumento de {random.randint(5, 20)}% nas vendas comparado ao mês anterior.  
2. Despesas: Os custos operacionais foram reduzidos em {random.randint(2, 10)}%.  
3. Metas: {random.randint(80, 95)}% das metas foram cumpridas.  
Conclusão: A empresa está em trajetória de crescimento, com recomendações para otimizar processos logísticos."""

# Função para gerar e-mail (rótulo 0)
def generate_email():
    remetente = random.choice(nomes)
    destinatario = random.choice(nomes)
    data = random.choice(datas)
    assunto = fake.catch_phrase()
    return f"""De: {remetente} <{fake.email()}>  
Para: {destinatario} <{fake.email()}>  
Assunto: {assunto}  
Data: {data}  
Prezado(a) {destinatario},  
Gostaríamos de informar que o projeto está avançando conforme planejado. Por favor, envie os documentos solicitados até o final da semana.  
Atenciosamente,  
{remetente}"""

# Função para gerar outros documentos (rótulo 0)
def gerar_outro():
    tipos_documentos = [
        "financiamento", "contrato_trabalho", "fatura", "relatorio", "email"
    ]
    tipo_documento = random.choice(tipos_documentos)
    if tipo_documento == "financiamento":
        return {"texto": generate_financing_contract(), "rotulo": 0}
    elif tipo_documento == "contrato_trabalho":
        return {"texto": generate_employment_contract(), "rotulo": 0}
    elif tipo_documento == "fatura":
        return {"texto": generate_invoice(), "rotulo": 0}
    elif tipo_documento == "relatorio":
        return {"texto": generate_report(), "rotulo": 0}
    elif tipo_documento == "email":
        return {"texto": generate_email(), "rotulo": 0}

# Gerar dataset
dataset = []
# 2500 exemplos de rótulo 1 (cessão de consórcio)
for _ in range(2500):
    dataset.append(gerar_cessao_consorcio())
# 2500 exemplos de rótulo 0 (outros documentos)
for _ in range(2500):
    dataset.append(gerar_outro())

# Embaralhar dataset
random.shuffle(dataset)

# Verificar balanceamento
print(f"Total de exemplos: {len(dataset)}")
print(f"Rótulo 0 (outros documentos): {sum(1 for d in dataset if d['rotulo'] == 0)}")
print(f"Rótulo 1 (cessão de consórcio): {sum(1 for d in dataset if d['rotulo'] == 1)}")

# Salvar em JSON
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset com {NUM_EXAMPLES} registros salvo em: {OUTPUT_PATH}")