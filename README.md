# Dashboard de Análise de Estresse Acadêmico

Este repositório contém um dashboard desenvolvido em Python com Streamlit para análise de estresse acadêmico.

## Funcionalidades
- Coleta e limpeza de dados.
- Dashboard interativo com filtros e gráficos.
- Visualizações: distribuições, boxplots e scatter plots.

## Estrutura do projeto
```
projeto-estresse-academico/
├── data/
│   ├── academic-stress-level.csv
│   └── estresse_academico_univesp.csv
│
├── pages/
│   ├── kaggler.py
│   └── univesp.py
│
├── dashboard.py
├── requirements.txt
├── README.md
│
└── .gitignore
```

## Tecnologias
- Python 3.8+
- Streamlit, Pandas, Plotly, NumPy, Matplotlib, Seaborn, Scikit-learn

## Como executar
1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/estresse-academico-dashboard.git
cd estresse-academico-dashboard
```
2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Instale dependências:
```bash
pip install -r requirements.txt
```
4. Execute o dashboard:
```bash
streamlit run dashboard.py
```

## Dados e processamento
- Arquivo principal: `data/estresse_academico_univesp.csv`
- Pré-processamento realizado em `preprocess.py`:
    - Mapeamento de categorias.
    - Tratamento de valores ausentes.

## Resultados
- Visualizações interativas para análise de fatores associados ao estresse.
- Relatório em `reports/resumo_resultados.md`.

## Deploy
- Deploy no Streamlit Cloud ou uso de Docker.

## Licença
Projeto sob licença MIT.

## Contribuição
Contribuições são bem-vindas via pull requests.

## Contato
email@example.com
