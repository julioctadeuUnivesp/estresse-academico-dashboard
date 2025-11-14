import streamlit as st


# Sidebar
st.sidebar.title("Navegação")
st.sidebar.info("Use o menu acima e aguarde o carregamento para navegar entre as páginas do dashboard.")

# Configuração da página
st.set_page_config(
    page_title="Dashboard Estresse Acadêmico",
    page_icon="📊",
    layout="centered"
)

# CSS personalizado aprimorado
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding: 10px 16px;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Título e descrição
# Título principal
st.markdown('<h1 class="main-header">🎓 Dashboard sobre Estresse Acadêmico</h1>', unsafe_allow_html=True)
st.markdown("""
Bem-vindo ao **Dashboard Estresse Acadêmico**!  
Aqui você poderá explorar dados, visualizar gráficos e obter insights dos níveis de estresse acadêmico.  
""")
st.markdown("""
Foram utilizado dados coletados de diversas fontes para proporcionar uma análise abrangente.""")
st.markdown("""
A fonte de dados **KAGGLE** disponivel em [*Student Academic Stress Real World Dataset*](https://www.kaggle.com/datasets/poushal02/student-academic-stress-real-world-dataset) foi utilizada na pesquisa.
""")
st.markdown("""
A fonte de dados **UNIVESP** foi levantada por questionários realizados aos discentes da própria universidade.
""")

st.markdown(" Use o menu lateral para navegar entre as páginas.")

# Botão de navegação
if st.button("Explorar Dados"):
    st.write("Clique no menu lateral para começar!")
else:
    st.info("Use o menu lateral para acessar as funcionalidades.")

# Rodapé
st.markdown("---")