# 📊 Dashboard para Análise de Estresse Acadêmico

Este projeto utiliza Python para análise e visualização de dados sobre estresse acadêmico em estudantes universitários e corpo discente da UNIVESP.

## 📋 Sobre o Projeto

Análise de fatores predisponentes ao estresse acadêmico em estudantes de graduação da UNIVESP, com base em dados coletados através de formulário estruturado.


## Funcionalidades
- Coleta e padronização de dados.
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

### Bibliotecas Python
| Biblioteca | Versão | Finalidade |
|------------|--------|------------|
| **Pandas** | >=1.5.0 | Manipulação e análise de dados |
| **Streamlit** | >=1.22.0 | Criação de dashboard interativo |
| **Plotly** | >=5.13.0 | Visualizações interativas |
| **NumPy** | >=1.23.0 | Operações numéricas |
| **Matplotlib** | >=3.7.0 | Visualizações estáticas |

### Descrição das Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal utilizada para o desenvolvimento do projeto, incluindo o processamento de dados e a criação do dashboard.
    - Exemplo: Scripts como `dashboard.py` e `preprocess.py` foram escritos em Python para manipular os dados e gerar as visualizações.

- **Streamlit**: Framework usado para construir o dashboard interativo, permitindo a visualização e interação com os dados de forma simples e eficiente.
    - Exemplo: O arquivo `dashboard.py` utiliza Streamlit para criar a interface do usuário com gráficos e filtros interativos.

- **Pandas**: Biblioteca utilizada para manipulação e análise de dados, incluindo a limpeza e transformação dos datasets.
    - Exemplo: No script `preprocess.py`, o Pandas é usado para tratar valores ausentes e mapear categorias nos datasets.

- **Plotly**: Ferramenta para criar gráficos interativos e visualizações dinâmicas no dashboard.
    - Exemplo: Gráficos como scatter plots e boxplots no dashboard foram gerados com Plotly para maior interatividade.

- **NumPy**: Biblioteca usada para operações numéricas e manipulação de arrays, auxiliando no pré-processamento dos dados.
    - Exemplo: Operações matemáticas e manipulação de arrays no script `preprocess.py` foram realizadas com NumPy.

- **Matplotlib**: Complementa as visualizações com gráficos estáticos e auxiliares durante a análise exploratória dos dados.
    - Exemplo: Durante a análise inicial dos dados, gráficos estáticos foram criados com Matplotlib para identificar padrões e tendências.

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

## Resultados
- Visualizações interativas para análise de fatores associados ao estresse.

## Deploy
- Deploy no Streamlit Cloud ou uso de Docker.

## Licença
Projeto sob licença MIT.

## Contribuição
Contribuições são bem-vindas via pull requests.

## Contato
email@example.com
