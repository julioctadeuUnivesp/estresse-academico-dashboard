# 📊 Dashboard para Análise de Estresse Acadêmico

Este projeto utiliza Python para análise e visualização de dados sobre estresse acadêmico em estudantes de diferentes níveis educacionais.

## 📋 Sobre o Projeto

Análise de fatores predisponentes ao estresse acadêmico em estudantes do ensino médio, graduação e pós-graduação, com base em dados coletados através de formulário estruturado.

## 🛠️ Tecnologias Utilizadas

### Bibliotecas Python
| Biblioteca | Versão | Finalidade |
|------------|--------|------------|
| **Pandas** | >=1.5.0 | Manipulação e análise de dados |
| **Streamlit** | >=1.22.0 | Criação de dashboard interativo |
| **Plotly** | >=5.13.0 | Visualizações interativas |
| **NumPy** | >=1.23.0 | Operações numéricas |
| **Matplotlib** | >=3.7.0 | Visualizações estáticas |
| **Seaborn** | >=0.12.0 | Visualizações estatísticas |
| **Scikit-learn** | >=1.2.0 | Machine learning (opcional) |

## 📁 Estrutura do Dataset

O dataset contém as seguintes colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| **Estágio Acadêmico** | Categórico | Nível educacional do aluno |
| **Pressão dos Colegas** | Numérico (1-5) | Influência dos pares |
| **Pressão Acadêmica de Casa** | Numérico (1-5) | Expectativas familiares |
| **Ambiente de Estudo** | Categórico | Qualidade do ambiente |
| **Estratégia de Enfrentamento** | Categórico | Métodos de coping |
| **Maus Hábitos** | Texto | Hábitos negativos |
| **Competição Acadêmica** | Numérico (1-5) | Intensidade competitiva |
| **Índice de Estresse** | Numérico (1-5) | Variável alvo |

## 🚀 Como Executar

### Pré-requisitos
```bash
python >=3.8
pip >=21.0
```

### Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/estresse-academico-dashboard.git
cd estresse-academico-dashboard

# Crie um ambiente virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Executar o Dashboard
```bash
streamlit run dashboard.py
```

## 📊 Funcionalidades do Dashboard

### ✅ Filtros Interativos
- Filtro por estágio acadêmico
- Filtro por faixa de estresse
- Filtro por pressão dos colegas
- Filtro por ambiente de estudo

### 📈 Visualizações Incluídas
- **Heatmap de Correlação**: Relação entre variáveis
- **Distribuição de Estresse**: Por nível educacional
- **Boxplots Comparativos**: Entre diferentes fatores
- **Gráfico de Dispersão**: Pressão vs. Estresse
- **Análise de Estratégias de Coping**: Por grupo

## 🎯 Exemplo de Código Principal

```python
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(page_title="Dashboard Estresse Acadêmico", layout="wide")

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_csv('dados_estresse_academico.csv')

df = load_data()

# Sidebar com filtros
st.sidebar.header("Filtros")
nivel_educacional = st.sidebar.multiselect(
    "Nível Educacional",
    options=df['Estágio Acadêmico'].unique(),
    default=df['Estágio Acadêmico'].unique()
)

# Aplicar filtros
df_filtrado = df[df['Estágio Acadêmico'].isin(nivel_educacional)]

# Layout principal
st.title("📊 Dashboard de Análise de Estresse Acadêmico")

# Métricas principais
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Estudantes", len(df_filtrado))
with col2:
    st.metric("Estresse Médio", f"{df_filtrado['Índice de Estresse'].mean():.2f}")
with col3:
    st.metric("Níveis Representados", df_filtrado['Estágio Acadêmico'].nunique())

# Gráficos
fig = px.box(df_filtrado, x='Estágio Acadêmico', y='Índice de Estresse', 
             title="Distribuição de Estresse por Nível Educacional")
st.plotly_chart(fig, use_container_width=True)
```

## 📁 Estrutura do Projeto

```
projeto-estresse-academico/
│
├── data/
│   ├── academic-stress-level.csv
│   └── estresse_academico_univesp.csv
│
├── dashboard.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔧 Processamento de Dados

```python
def preprocess_data(df):
    """Função de pré-processamento dos dados"""
    # Converter categorias
    mapping = {
        'noisy': 'Barulhento',
        'peaceful': 'Tranquilo', 
        'disrupted': 'Perturbado'
    }
    df['Ambiente de Estudo'] = df['Ambiente de Estudo'].map(mapping)
    
    # Tratar valores missing
    df = df.dropna()
    
    return df
```

## 📈 Exemplos de Análise

### Correlação entre Variáveis
```python
# Matriz de correlação
correlation_matrix = df_filtrado.corr()
fig = px.imshow(correlation_matrix, title="Matriz de Correlação")
st.plotly_chart(fig)
```

### Análise por Estratégia de Coping
```python
coping_analysis = df_filtrado.groupby('Estratégia de Enfrentamento')['Índice de Estresse'].mean()
fig = px.bar(coping_analysis, title="Estresse Médio por Estratégia de Enfrentamento")
st.plotly_chart(fig)
```

## 🚀 Deploy

### Deploy no Streamlit Cloud
1. Faça push do código para o GitHub
2. Acesse [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecte seu repositório
4. Configure o arquivo principal como `dashboard.py`

### Deploy Local com Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📧 Contato

Equipe de Análise de Dados - [email@example.com](mailto:email@example.com)

## 🔗 Links Úteis

- [Documentação Streamlit](https://docs.streamlit.io/)
- [Documentação Plotly](https://plotly.com/python/)
- [Documentação Pandas](https://pandas.pydata.org/docs/)
- [Dataset no Kaggle](https://www.kaggle.com/datasets/your-dataset)

---

**Nota**: Este projeto foi desenvolvido para fins acadêmicos como parte do Projeto Integrador da UNIVESP.
