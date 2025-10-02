# treinador-neural


pip freeze > requirements.txt
pip install -r requirements.txt

sha256sum data/contratos_cessao_5000.json
Get-FileHash data\contratos_cessao_5000.json -Algorithm SHA256
pip show torch transformers onnxruntime pandas  







# Exibir métricas
print("\n--- Resultados no Conjunto de Teste ---")
print("F1-score:", f1_score(y_true, y_pred, average='weighted'))
print("Precisão:", precision_score(y_true, y_pred, average='weighted'))
print("Recall:", recall_score(y_true, y_pred, average='weighted'))
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:")
print(f"  Verdadeiro Negativo (Classe 0 correta): {cm[0,0]}")
print(f"  Falso Positivo (Classe 0 errada): {cm[0,1]}")
print(f"  Falso Negativo (Classe 1 errada): {cm[1,0]}")
print(f"  Verdadeiro Positivo (Classe 1 correta): {cm[1,1]}")

# Salvar resultados em CSV
import pandas as pd


Analise Visual:

import pandas as pd
df = pd.read_csv("model/bert_finetuned/avaliacao_teste.csv")
print(df[df["true_label"] == 1][["texto_inicial", "prob_1", "pred_label"]])


# treinador-neural


Analise este repositório que contém a infraestrutura de ECS usando Terraform e GitHub Actions.

1. Reconhecimento do projeto:
   - Identifique a stack usada (ex: Terraform, AWS ECS, VPC, autoscaling, etc.).
   - Mapeie a estrutura de pastas e arquivos do repositório.
   - Entenda como os workflows de GitHub Actions estão configurados atualmente.
   - Liste quais ferramentas de validação ou testes já estão implementadas (se houver).

2. Proposta de TAaC (Test as a Code):
   - Sugira como aplicar práticas de Test as a Code ao projeto.
   - Inclua ferramentas para validação sintática, lint, segurança/compliance (terraform validate, fmt, tflint, tfsec, checkov).
   - Sugira como estruturar testes unitários e de integração (Terratest em Go ou Pytest com boto3).
   - Proponha smoke tests para ECS (ex: subir uma task/service e validar health check).
   - Mostre como versionar e organizar esses testes dentro do repositório.
   - Explique como integrar tudo em pipelines do GitHub Actions (exemplo de YAML).
   - Recomende métricas, relatórios e como lidar com múltiplos ambientes (dev, staging, prod).

3. Exemplos práticos:
   - Código de teste em Terratest ou Pytest.
   - Snippets de pipeline GitHub Actions com jobs de lint, validate, segurança e integração.



Quero que você analise meu repositório de infraestrutura de ECS, que usa Terraform e pipelines em GitHub Actions, 
e proponha uma abordagem de Test as a Code (TAaC).

Contexto:
- O repositório gerencia clusters ECS, services, task definitions, networking e autoscaling.
- IaC está em Terraform.
- O CI/CD é feito em GitHub Actions.
- O objetivo é garantir qualidade, segurança e confiabilidade da infraestrutura antes de qualquer deploy.

O que espero da sua análise:
1. Estratégia de testes para IaC em Terraform:
   - Lint e validação sintática (`terraform fmt`, `terraform validate`).
   - Testes estáticos de segurança e compliance (Checkov, TFLint, tfsec).
   - Testes unitários de módulos (Terratest em Go, Pytest com boto3).
   - Testes de integração em ambiente temporário (ex: LocalStack ou AWS real com cleanup).
2. Estrutura de pastas e organização dos testes dentro do repositório.
3. Exemplo de pipeline em GitHub Actions (YAML) que aplique o TAaC:
   - Jobs rápidos (lint, validate).
   - Jobs de segurança/compliance.
   - Jobs de testes de integração/smoke em ECS (ex: rodar task e validar health check).
4. Recomendações de métricas e relatórios (ex: cobertura de testes de módulos, resultados de segurança, drift detection).
5. Estratégia para múltiplos ambientes (dev, staging, prod) usando workspaces do Terraform e GitHub Actions com ambientes.
6. Sugestões de como evoluir a prática de TAaC para suportar escalabilidade e reuso em outros repositórios de infraestrutura.

Inclua exemplos práticos de:
- Código de teste (ex: Terratest em Go para validar criação de ECS Service).
- Snippets de pipeline GitHub Actions em YAML.

