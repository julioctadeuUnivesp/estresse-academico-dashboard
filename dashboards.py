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
    return pd.read_csv('academic-stress-level.csv')

df = load_data()

# Sidebar com filtros
st.sidebar.header("Filtros")
nivel_educacional = st.sidebar.multiselect(
    "Nível Educacional",
    options=df['Your Academic Stage'].unique(),
    default=df['Your Academic Stage'].unique()
)

# Aplicar filtros
df_filtrado = df[df['Your Academic Stage'].isin(nivel_educacional)]

# Layout principal
st.title("📊 Dashboard de Análise de Estresse Acadêmico")

# Métricas principais
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Estudantes", len(df_filtrado))
with col2:
    st.metric("Estresse Médio", f"{df_filtrado['Peer pressure'].mean():.2f}")
with col3:
    st.metric("Níveis Representados", df_filtrado['Your Academic Stage'].nunique())

# Gráficos
fig = px.box(df_filtrado, x='Your Academic Stage', y='Peer pressure', 
             title="Distribuição de Estresse por Nível Educacional")
st.plotly_chart(fig, use_container_width=True)