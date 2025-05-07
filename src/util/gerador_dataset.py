import json
import random
from pathlib import Path
import faker

# Configurações
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "contratos_reais_2000.json"
NUM_EXAMPLES = 5000  # 1000 rótulo 1, 1000 rótulo 0 (divididos entre 6 tipos)

# Inicializar o Faker para gerar dados fictícios
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

# Texto base para rótulo 1 (contrato de consórcio)
CONSORCIO_TEXT = """Aditamento ao Contrato de Participação em Grupo de Consórcio de Bem Móvel/Imóvel -— Cessão de Direitos e Obrigações 1. Administradora/Credora Fiduciária Itaú Adm de Consórcio Ltda com sede na Alameda Pedro Calil, 43, Centro, Poá, SP, inscrita no CNPJ sob o nº 00.000.776-0001-01. 2. Cedente . 2.1 Nome 2.2 CPFÍICNPJ MARIA DA CONSOLAÇÃO FREITAS 009.678.466-09 3. Cessionário 3.1 Nome 3.2 CPFÍCNPJ Fundo de Investimentos em Direitos Creditórios Consorciei II 43.237.790/0001-36 3.3 EndereçoiSede 3.4 Dados da Conta Corrente Rua Gomes de Carvalho, nº. 1195, 4º andar, São Paulo/SP Agência Conta - DAC 3.5. Telefones do cessionário: (11) 3500-7238 4. Dados do Contrato 4.1 Grupo 4.2 Cota 4.3 Percentual Pago 4.4 Percentual a Vencer 20280 283 38.71% 61.29% 5.Modo de Pagamento 6.Tarifa de Aditamento Contratual 5.1 (X) documento de cobrança R$ 650,00 5.2( ) débito na conta corrente indicada no subitem A empresa indicada no item 1, doravante designada Administradora, e as pessoas qualificadas nos itens 2 e 3, designadas respectivamente, Cedente e Cessionário, aditam o Contrato de Participação em Grupo de Consórcio de Bem Móvelilmóvel (“Contrato de Adesão”) indicado no item 4, de acordo com as cláusulas que seguem. 7. Objeto - O Cedente, autorizado pela Administradora, transfere ao Cessionário todos direitos e obrigações previstos no Contrato de Adesão que regula o Grupo/Cota de consórcio apontados no item 4. 7.1 0 CESSIONÁRIO DECLARA QUE RECEBEU CÓPIA DO CONTRATO DE ADESÃO, O LEU PREVIAMENTE. NÃO TENDO QUALQUER DÚVIDA EM RELAÇÃO AO QUE ALI RESTOU DISPOSTO. CONCORDA COM TODAS AS CLÁUSULAS E CONDIÇÕES DO CONTRATO DE ADESÃO E. ATRAVÉS DESTE INSTRUMENTO, ASSUME TODAS AS OBRIGAÇÕES NELE PREVISTAS, EM ESPECIAL AS DE NATUREZA FINANCEIRA. 8. Modo de Pagamento - O Cessionário fará os pagamentos na forma indicada no item 5. 8.1. Sendo assinalado o subitem 5.2, o Cessionário pagará todos os valores devidos em decorrência do Contrato de Adesão mediante débito em sua conta corrente mantida junto ao Itaú Unibanco S.A., indicada no subitem 3.4, que deverá ter saldo disponível suficiente. O Itaú Unibanco S.A., desde já autorizado a efetuar o débito, entregará o respectivo valor à Administradora. 8.2 A insuficiência de saldo na conta corrente apontada no subitem 3.4 configurará atraso no pagamento. 8.3 Sendo indicado o subitem 5.1, o Cessionário fará todos os pagamentos por meio de documento de cobrança (camê ou equivalente) a ser emitido pelo Itaú Unibanco S.A. e enviado para o endereço indicado no subitem 3.3. Se o Cessionário não receber o documento de cobrança em tempo hábil para efetuar o pagamento, deverá comunicar o fato à Administradora, que o instruirá a respeito de como proceder. Em nenhuma hipótese o não recebimento do documento de cobrança eximirá o Cessionário do pagamento. 8.4 O Cessionário está ciente de que. havendo atraso no pagamento de qualquer parcela. ficará impedido de concorrer aos sorteios e de ofertar lances, sem prejuizo das demais sanções previstas no Contrato de Adesão. 9. Tolerância — A tolerância das partes quanto ao descumprimento de qualquer obrigação pela outra parte não significará renúncia ao direito de exigir o cumprimento da obrigação, nem perdão, nem alteração do que foi aqui ajustado. 10.Tarifa - O Cessionário pagará a taxa indicada no item 6 e prevista no Contrato de Adesão, em razão desta SEX - Baik Otiie IICVO --- Página 2 --- DocuSign Envelope ID: 082146A0-07F7-4AF0-99B6-591DD8A89B1F cessão de direitos e obrigações. 11. O CESSIONÁRIO, NESTE ATO, CONFERE PODERES ESPECIAIS À ADMINISTRADORA, NA QUALIDADE DE GESTORA DOS NEGÓCIOS DO GRUPO E MANDATÁRIA DOS SEUS INTERESSES E DIREITOS, PARA, EM CARÁTER IRREVOGÁVEL E IRRETRATÁVEL, (I) TOMAR TODAS AS PROVIDÊNCIAS NECESSÁRIAS À ADMINISTRAÇÃO DO GRUPO. INCLUSIVE RECEBER E DAR QUITAÇÃO, EFETUAR PAGAMENTOS E CONSTITUIR ADVOGADOS PARA A DEFESA DOS INTERESSES DO GRUPO; (Il) REPRESENTÁ-LO PERANTE OUTROS CONSORCIADOS, TERCEIROS, ÓRGÃOS GOVERNAMENTAIS E EMPRESAS SEGURADORAS PARA A CONTRATAÇÃO DOS SEGUROS PREVISTOS NO CONTRATO DE ADESÃO; E (Ill) REPRESENTÁ-LO NAS ASSEMBLEIAS GERAIS ORDINÁRIAS EM QUE NÃO ESTIVER PRESENTE, INCLUSIVE VOTANDO AS MATÉRIAS DA ORDEM DO DIA. 12. O processo de cessão de cota será automaticamente cancelado se estiver em andamento o registro do contrato perante o cartório de registro de imóveis. 13. Cláusulas inalteradas - Permanecem em vigor as disposições do Contrato de Adesão aditado não expressamente alteradas por este Aditamento. 14. Foro - Fica eleito o Foro da Comarca do local da assinatura deste Aditamento, podendo a parte que promover a ação optar pelo Foro do domicílio do Cessionário. 15. Solução Amigável de Conflitos - Para a solução amigável de eventuais conflitos relacionados a este Aditamento, o Cessionário poderá dirigir seu pedido ou reclamação à sua agência Itaú Unibanco S/A. A Administradora coloca ainda à disposição do Cessionário o SAC - Itaú (0800 728 0728). o SAC - Itaú exclusivo ao deficiente auditivo (0800 722 1722) e o Fale Conosco (wwnw.itau.com.br). Se não for solucionado o conflito, o Cessionário poderá recorrer à Ouvidoria Corporativa Itaú (0800 570 0011, em dias úteis das 9h às 18h, Caixa Postal 67.600, CEP 03162-971). E por estarem de acordo, assinam o presente em 03 (três) vias de igual teor e forma, na presença de 02 (duas) testemunhas. Local e Data: São Paulo, 03 de janeiro de 2024 Li (LEMOS) ESTE ADITAMENTO PREVIAMENTE E NÃO TENHO (TEMOS) DÚVIDA SOBRE QUALQUER UMA DE SUAS CLÁUSULAS. DocuSigned by Olsagndae Calumans dog rrta Cessionári (04 Cedente fundo de Investimentos em Direitos Creditórios MARIA DA CONSOLAÇÃO FREITAS onsorcie 3 o) Rataei inocençio À Biencou: RT L5I062.2 cos poa re e Adminiskadora Devedor Solidário a) o) Nome: Nome: CPF CPF Testemunhas a) D) Nome: Nome CPF: CPF: SEX - Baik Otiie IICVO"""

# Função para gerar contrato de cessão (rótulo 1) com variações
def gerar_cessao():
    cedente = random.choice(nomes)
    cessionario = random.choice(fundos)
    cota = random.randint(100, 9999)
    grupo = random.randint(10000, 99999)
    tarifa = random.randint(500, 2000)
    saldo = random.randint(40, 80)
    admin = random.choice(administradoras)
    bem = random.choice(["Móvel", "Imóvel"])
    data = random.choice(datas)
    cidade = random.choice(cidades)
    modified_text = CONSORCIO_TEXT.replace("MARIA DA CONSOLAÇÃO FREITAS", cedente) \
                                 .replace("009.678.466-09", fake.cpf()) \
                                 .replace("Fundo de Investimentos em Direitos Creditórios Consorciei II", cessionario) \
                                 .replace("43.237.790/0001-36", fake.cnpj()) \
                                 .replace("R$ 650,00", f"R${tarifa},00") \
                                 .replace("20280", str(grupo)) \
                                 .replace("283", str(cota)) \
                                 .replace("38.71%", f"{saldo}%") \
                                 .replace("61.29%", f"{100-saldo}%") \
                                 .replace("Itaú Adm de Consórcio Ltda", admin) \
                                 .replace("São Paulo, 03 de janeiro de 2024", f"{cidade}, {data}")
    return {"texto": modified_text, "rotulo": 1}

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

# Função para gerar um contrato de venda de criptomoedas (rótulo 0)
def generate_crypto_sale_contract():
    vendedor = random.choice(nomes)
    comprador = random.choice(nomes)
    cpf_vendedor = fake.cpf()
    cpf_comprador = fake.cpf()
    criptomoeda = random.choice(["Bitcoin (BTC)", "Ethereum (ETH)", "Cardano (ADA)", "Solana (SOL)"])
    quantidade = round(random.uniform(0.1, 5.0), 4)
    valor = round(random.uniform(5000, 50000), 2)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_financiador = fake.address().replace('\n', ', ')
    endereco_financiado = fake.address().replace('\n', ', ')
    return f"""Contrato de Venda de Criptomoedas nº {random.randint(1000, 9999)}  
1. Partes  
1.1. Vendedor: {vendedor}, CPF nº {cpf_vendedor}, residente em {endereco_financiador}.  
1.2. Comprador: {comprador}, CPF nº {cpf_comprador}, residente em {endereco_financiado}.  
2. Objeto do Contrato  
O Vendedor compromete-se a vender ao Comprador a quantidade de {quantidade} {criptomoeda}, pelo valor total de R${valor}.  
3. Condições de Pagamento  
3.1. O pagamento será realizado via transferência bancária para a conta indicada pelo Vendedor no Anexo I, até a data de {data}.  
3.2. Após a confirmação do pagamento, o Vendedor transferirá as criptomoedas para a carteira digital do Comprador, indicada no Anexo II.  
4. Declarações  
4.1. O Vendedor declara ser o legítimo proprietário das criptomoedas e que as mesmas estão livres de quaisquer ônus ou restrições.  
4.2. O Comprador declara ter ciência dos riscos associados ao investimento em criptomoedas.  
5. Foro  
Fica eleito o foro da comarca de {cidade} para dirimir quaisquer dúvidas ou litígios decorrentes deste contrato.  
Local e Data: {cidade}, {data}.  
Assinaturas:  
Vendedor: ___________________________  
Comprador: ___________________________  
Testemunhas:  
1. Nome: {random.choice(nomes)} CPF: {fake.cpf()}  
2. Nome: {random.choice(nomes)} CPF: {fake.cpf()}"""

# Função para gerar um contrato de compra de ações (rótulo 0)
def generate_stock_purchase_contract():
    vendedor = random.choice(nomes)
    comprador = random.choice(nomes)
    cpf_vendedor = fake.cpf()
    cpf_comprador = fake.cpf()
    empresa = fake.company()
    quantidade = random.randint(100, 1000)
    valor_por_acao = round(random.uniform(10, 100), 2)
    valor_total = quantidade * valor_por_acao
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_financiador = fake.address().replace('\n', ', ')
    endereco_financiado = fake.address().replace('\n', ', ')
    return f"""Contrato de Compra e Venda de Ações nº {random.randint(1000, 9999)}  
1. Partes  
1.1. Vendedor: {vendedor}, CPF nº {cpf_vendedor}, residente em {endereco_financiador}.  
1.2. Comprador: {comprador}, CPF nº {cpf_comprador}, residente em {endereco_financiado}.  
2. Objeto do Contrato  
O Vendedor compromete-se a vender ao Comprador {quantidade} ações ordinárias da empresa {empresa}, pelo valor unitário de R${valor_por_acao}, totalizando R${valor_total}.  
3. Condições de Pagamento  
3.1. O pagamento será realizado via transferência bancária para a conta indicada pelo Vendedor no Anexo I, até a data de {data}.  
3.2. Após a confirmação do pagamento, as ações serão transferidas para a conta de custódia do Comprador, indicada no Anexo II.  
4. Declarações  
4.1. O Vendedor declara ser o legítimo proprietário das ações e que as mesmas estão livres de quaisquer ônus ou restrições.  
4.2. O Comprador declara ter ciência dos riscos associados ao investimento em ações.  
5. Foro  
Fica eleito o foro da comarca de {cidade} para dirimir quaisquer dúvidas ou litígios decorrentes deste contrato.  
Local e Data: {cidade}, {data}.  
Assinaturas:  
Vendedor: ___________________________  
Comprador: ___________________________  
Testemunhas:  
1. Nome: {random.choice(nomes)} CPF: {fake.cpf()}  
2. Nome: {random.choice(nomes)} CPF: {fake.cpf()}"""

# Função para gerar um contrato de leasing de automóvel (rótulo 0)
def generate_leasing_contract():
    arrendador = fake.company()
    arrendatario = random.choice(nomes)
    cnpj_arrendador = fake.cnpj()
    cpf_arrendatario = fake.cpf()
    veiculo = random.choice(veiculos)
    valor = round(random.uniform(2000, 5000), 2)
    parcelas = random.randint(24, 48)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_financiador = fake.address().replace('\n', ', ')
    endereco_financiado = fake.address().replace('\n', ', ')
    return f"""Contrato de Leasing de Automóvel nº {random.randint(1000, 9999)}  
1. Partes  
1.1. Arrendador: {arrendador}, com sede em {endereco_financiador}, inscrito no CNPJ sob o nº {cnpj_arrendador}.  
1.2. Arrendatário: {arrendatario}, CPF nº {cpf_arrendatario}, residente em {endereco_financiado}.  
2. Objeto do Contrato  
O Arrendador concede ao Arrendatário o uso do veículo {veiculo}, pelo prazo de {parcelas} meses, mediante o pagamento de parcelas mensais no valor de R${valor}.  
3. Condições de Pagamento  
3.1. As parcelas serão pagas via boleto bancário, com vencimento no dia 5 de cada mês, iniciando em {data}.  
3.2. O Arrendatário terá a opção de compra do veículo ao final do contrato, pelo valor residual estipulado no Anexo I.  
4. Obrigações do Arrendatário  
4.1. O Arrendatário será responsável pela manutenção e seguro do veículo durante o período de leasing.  
4.2. O veículo deverá ser devolvido ao Arrendador ao final do contrato, caso a opção de compra não seja exercida.  
5. Inadimplemento  
Em caso de atraso no pagamento, será cobrada multa de 2% e juros de mora de 1% ao mês sobre o valor da parcela em atraso.  
6. Foro  
Fica eleito o foro da comarca de {cidade} para dirimir quaisquer dúvidas ou litígios decorrentes deste contrato.  
Local e Data: {cidade}, {data}.  
Assinaturas:  
Arrendador: ___________________________  
Arrendatário: ___________________________  
Testemunhas:  
1. Nome: {random.choice(nomes)} CPF: {fake.cpf()}  
2. Nome: {random.choice(nomes)} CPF: {fake.cpf()}"""

# Função para gerar um contrato de confidencialidade de participação em projeto (rótulo 0)
def generate_confidentiality_contract():
    empresa = fake.company()
    participante = random.choice(nomes)
    cnpj_empresa = fake.cnpj()
    cpf_participante = fake.cpf()
    projeto = fake.catch_phrase()
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_financiador = fake.address().replace('\n', ', ')
    endereco_financiado = fake.address().replace('\n', ', ')
    return f"""Acordo de Confidencialidade de Participação em Projeto nº {random.randint(1000, 9999)}  
1. Partes  
1.1. Empresa: {empresa}, com sede em {endereco_financiador}, inscrito no CNPJ sob o nº {cnpj_empresa}.  
1.2. Participante: {participante}, CPF nº {cpf_participante}, residente em {endereco_financiado}.  
2. Objeto do Acordo  
Este acordo tem como objetivo garantir a confidencialidade das informações relacionadas ao Projeto {projeto}, no qual o Participante atuará como consultor/colaborador.  
3. Obrigações de Confidencialidade  
3.1. O Participante compromete-se a não divulgar, reproduzir ou utilizar as informações confidenciais do Projeto para qualquer finalidade que não seja a execução do mesmo.  
3.2. As informações confidenciais incluem, mas não se limitam a, dados técnicos, financeiros, comerciais e estratégicos do Projeto.  
4. Prazo de Confidencialidade  
As obrigações de confidencialidade permanecerão em vigor por {random.randint(1, 5)} anos após o término do Projeto.  
5. Penalidades  
O descumprimento deste acordo sujeitará o Participante a indenizações por perdas e danos, além de medidas judiciais cabíveis.  
6. Foro  
Fica eleito o foro da comarca de {cidade} para dirimir quaisquer dúvidas ou litígios decorrentes deste acordo.  
Local e Data: {cidade}, {data}.  
Assinaturas:  
Empresa: ___________________________  
Participante: ___________________________  
Testemunhas:  
1. Nome: {random.choice(nomes)} CPF: {fake.cpf()}  
2. Nome: {random.choice(nomes)} CPF: {fake.cpf()}"""

# Função para gerar um contrato de trabalho (rótulo 0)
def generate_employment_contract():
    empregador = fake.company()
    empregado = random.choice(nomes)
    cnpj_empregador = fake.cnpj()
    cpf_empregado = fake.cpf()
    cargo = fake.job()
    salario = round(random.uniform(2000, 10000), 2)
    data = random.choice(datas)
    cidade = random.choice(cidades)
    endereco_financiador = fake.address().replace('\n', ', ')
    endereco_financiado = fake.address().replace('\n', ', ')
    return f"""Contrato de Trabalho nº {random.randint(1000, 9999)}  
1. Partes  
1.1. Empregador: {empregador}, com sede em {endereco_financiador}, inscrito no CNPJ sob o nº {cnpj_empregador}.  
1.2. Empregado: {empregado}, CPF nº {cpf_empregado}, residente em {endereco_financiado}.  
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

# Função para gerar outros contratos (rótulo 0)
def gerar_outro():
    tipos_contratos = [
        "financiamento", "venda_criptomoedas", "compra_acoes",
        "leasing_automovel", "confidencialidade", "contrato_trabalho"
    ]
    tipo_contrato = random.choice(tipos_contratos)

    if tipo_contrato == "financiamento":
        return {"texto": generate_financing_contract(), "rotulo": 0}
    elif tipo_contrato == "venda_criptomoedas":
        return {"texto": generate_crypto_sale_contract(), "rotulo": 0}
    elif tipo_contrato == "compra_acoes":
        return {"texto": generate_stock_purchase_contract(), "rotulo": 0}
    elif tipo_contrato == "leasing_automovel":
        return {"texto": generate_leasing_contract(), "rotulo": 0}
    elif tipo_contrato == "confidencialidade":
        return {"texto": generate_confidentiality_contract(), "rotulo": 0}
    elif tipo_contrato == "contrato_trabalho":
        return {"texto": generate_employment_contract(), "rotulo": 0}

# Gerar dataset
dataset = []
# 1000 exemplos de rótulo 1
for _ in range(2500):
    dataset.append(gerar_cessao())

# 1000 exemplos de rótulo 0 (distribuídos uniformemente entre os 6 tipos)
num_samples_per_type = 2500 // 6  # ~166 exemplos por tipo
for tipo in ["financiamento", "venda_criptomoedas", "compra_acoes", "leasing_automovel", "confidencialidade", "contrato_trabalho"]:
    for _ in range(num_samples_per_type):
        dataset.append(gerar_outro())

# Ajustar para garantir exatamente 1000 exemplos de rótulo 0
while sum(1 for d in dataset if d["rotulo"] == 0) < 1000:
    dataset.append(gerar_outro())

# Embaralhar dataset
random.shuffle(dataset)

# Salvar em JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset com {NUM_EXAMPLES} registros salvo em: {OUTPUT_PATH}")