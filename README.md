# Modelo de Previsão de Notas para Receitas

Este projeto contém um modelo de Machine Learning treinado para prever a nota (rating) que um usuário daria a uma determinada receita, com base em seu histórico e nas características da receita.

## Funcionalidades

-   **Previsão de Notas**: Estima a avaliação de um usuário para uma receita em uma escala de 0 a 5.
-   **Modelo Avançado**: Utiliza um `XGBoost Regressor`, um algoritmo robusto e eficiente para tarefas de regressão.
-   **Engenharia de Features**: O modelo leva em conta dezenas de características, incluindo:
    -   **Features de Popularidade da Receita**: Nota média e número de avaliações.
    -   **Features do Perfil do Usuário**: Nota média que o usuário costuma dar e sua atividade.
    -   **Features Temporais**: Ano de publicação e "idade" da receita.
    -   **Features de Conteúdo**: Tags, palavras-chave e informações nutricionais da receita.
-   **Script de Previsão**: Inclui o script `predict.py` para gerar previsões em lote a partir de um arquivo de interações.

## Estrutura de Pastas

Para que o projeto funcione corretamente, a estrutura de pastas deve ser a seguinte:

```
/
├── archive/
│   ├── RAW_recipes.csv
│   ├── interactions_train.csv
│   └── interactions_test.csv  <-- Arquivo usado para as previsões
│
├── models/
│   ├── modelo_xgboost_final.joblib  <-- Modelo de IA treinado
│   └── tags_vectorizer.joblib       <-- Vetorizador de tags salvo
│
├── venv/
├── predict.py                     <-- Script para executar as previsões
├── requirements.txt               <-- Lista de dependências
└── README.md                      <-- Este arquivo
```

## Pré-requisitos

-   Python 3.8 ou superior
-   Pip (gerenciador de pacotes do Python)

## Instalação

Siga os passos abaixo para configurar o ambiente e instalar as dependências necessárias.

**1. Clone ou baixe este repositório:**
Mova os arquivos para uma pasta de sua escolha.

**2. Crie um ambiente virtual:**
É uma boa prática isolar as dependências do projeto. Abra o terminal na pasta do projeto e execute:
```bash
python -m venv venv
```

**3. Ative o ambiente virtual:**
-   **No Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate
    ```
-   **No macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```
Seu terminal agora deve indicar que você está no ambiente `(venv)`.

**4. Instale as bibliotecas necessárias:**
Use o arquivo `requirements.txt` para instalar todas as dependências de uma vez.
```bash
pip install -r requirements.txt
```
*(Se você não tiver um arquivo `requirements.txt`, pode criá-lo com `pip freeze > requirements.txt` após instalar as bibliotecas manualmente: `pip install pandas numpy scikit-learn xgboost joblib`)*

## Como Usar

Com o ambiente configurado e os dados no lugar certo, você pode gerar as previsões facilmente.

**1. Verifique os dados de entrada:**
-   Garanta que a pasta `archive/` contém os arquivos `RAW_recipes.csv`, `interactions_train.csv` e o arquivo no qual você quer fazer as previsões (`interactions_test.csv`).
-   Garanta que a pasta `models/` contém os arquivos `modelo_xgboost_final.joblib` e `tags_vectorizer.joblib`.

**2. Execute o script de previsão:**
No terminal (com o ambiente virtual ativado), execute o seguinte comando:
```bash
python predict.py
```

**3. Verifique a saída:**
O script irá imprimir uma amostra das previsões no terminal e, o mais importante, criará um novo arquivo na raiz do projeto chamado **`predictions.csv`**.

## Arquivo de Saída (`predictions.csv`)

O arquivo gerado conterá as colunas do arquivo de entrada (`interactions_test.csv`) com uma coluna adicional, `predicted_rating`, que contém a nota prevista pelo modelo.

**Exemplo do conteúdo de `predictions.csv`:**

| user_id | recipe_id | date       | rating | predicted_rating |
| :------ | :-------- | :--------- | :----- | :--------------- |
| 8937    | 44551     | 2005-12-23 | 4.0    | 4.01             |
| 56680   | 126118    | 2006-10-07 | 4.0    | 3.98             |
| 349752  | 219596    | 2008-04-12 | 0.0    | 0.55             |
| 628951  | 82783     | 2007-11-13 | 2.0    | 2.15             |

*(Nota: os valores de `predicted_rating` são exemplos de um modelo sem vazamento de dados e podem variar).*