import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Estresse Acadêmico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('data/estresse_academico_univesp.csv', sep=',', encoding='utf-8')
    
    # Limpeza básica dos dados
    df.columns = [col.strip().replace('"', '') for col in df.columns]
    
    return df

df = load_data()

# Título principal
st.title("📊 Dashboard de Estresse Acadêmico - Univesp")
st.markdown("---")

# Filtros em formato de gráficos (pizza) no cabeçalho
st.markdown("### 🔍 Filtros (clique/selecione abaixo)")

col_a, col_b = st.columns(2)

# Gráfico de pizza - Eixo do curso
contagem_eixos = df['Qual o eixo do seu curso?'].dropna().value_counts()
fig_eixos_header = px.pie(
    names=contagem_eixos.index,
    values=contagem_eixos.values,
    color_discrete_sequence=px.colors.qualitative.Set3,
    title="Eixo do Curso"
)
fig_eixos_header.update_layout(margin=dict(t=40, b=10, l=10, r=10))
with col_a:
    st.plotly_chart(fig_eixos_header, use_container_width=True)
    # controle auxiliar (sincroniza o filtro com o gráfico)
    eixos_options = ['Todos'] + list(contagem_eixos.index)
    eixo_selecionado = st.selectbox("Filtrar por Eixo", eixos_options, index=0)

# Gráfico de pizza - Semestre
contagem_semestres = df['Qual semestre se encontra?'].dropna().value_counts()
fig_sem_header = px.pie(
    names=contagem_semestres.index,
    values=contagem_semestres.values,
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Semestre"
)

fig_sem_header.update_layout(margin=dict(t=40, b=10, l=10, r=10))
with col_b:
    st.plotly_chart(fig_sem_header, use_container_width=True)
    semestres_options = ['Todos'] + list(contagem_semestres.index)
    semestre_selecionado = st.selectbox("Filtrar por Semestre", semestres_options, index=0)

# Aplicar filtros
df_filtrado = df.copy()

if eixo_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Qual o eixo do seu curso?'] == eixo_selecionado]
if semestre_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Qual semestre se encontra?'] == semestre_selecionado]

# Métricas principais
st.subheader("📈 Métricas Gerais")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_participantes = len(df_filtrado)
    st.metric("Total de Participantes", total_participantes)

with col2:
    media_estresse = df_filtrado['Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'].mean()
    st.metric("Média de Estresse nas Provas", f"{media_estresse:.1f}/5")

with col3:
    perc_acompanhamento = (df_filtrado['Faz acompanhamento psicológico ou cuidado a saúde mental?'] == 'Sim').mean() * 100
    st.metric("Faz Acompanhamento Psicológico", f"{perc_acompanhamento:.1f}%")

with col4:
    perc_ambiente_estudo = (df_filtrado['Tem um ambiente separado para seus estudos?'] == 'Sim').mean() * 100
    st.metric("Tem Ambiente de Estudo", f"{perc_ambiente_estudo:.1f}%")

st.markdown("---")

# Primeira linha de gráficos
col1, col2 = st.columns(2)

with col1:
    # Distribuição de estresse nas provas
    st.subheader("📊 Distribuição do Nível de Estresse nas Provas")
    fig_estresse = px.histogram(
        df_filtrado, 
        x='Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?',
        nbins=5,
        color_discrete_sequence=['#FF6B6B']
    )
    fig_estresse.update_layout(
        xaxis_title="Nível de Estresse (1-5)",
        yaxis_title="Quantidade de Alunos",
        showlegend=False
    )
    st.plotly_chart(fig_estresse, use_container_width=True)

with col2:
    # Distribuição por eixo do curso
    st.subheader("🎓 Distribuição por Eixo do Curso")
    contagem_eixos = df_filtrado['Qual o eixo do seu curso?'].value_counts()
    fig_eixos = px.pie(
        values=contagem_eixos.values,
        names=contagem_eixos.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_eixos, use_container_width=True)

# Segunda linha de gráficos
col1, col2 = st.columns(2)

with col1:
    # Interferência do trabalho nos estudos
    st.subheader("💼 Interferência do Trabalho nos Estudos")
    contagem_trabalho = df_filtrado['Quanto sua rotina de trabalho interfere nos estudos?'].value_counts().sort_index()
    fig_trabalho = px.bar(
        x=contagem_trabalho.index,
        y=contagem_trabalho.values,
        color_discrete_sequence=['#4ECDC4']
    )
    fig_trabalho.update_layout(
        xaxis_title="Nível de Interferência (1-5)",
        yaxis_title="Quantidade de Alunos"
    )
    st.plotly_chart(fig_trabalho, use_container_width=True)

with col2:
    # Interferência da rotina de casa
    st.subheader("🏠 Interferência da Rotina de Casa")
    contagem_casa = df_filtrado['Quanto sua rotina de casa interfere nos estudos?'].value_counts().sort_index()
    fig_casa = px.bar(
        x=contagem_casa.index,
        y=contagem_casa.values,
        color_discrete_sequence=['#45B7D1']
    )
    fig_casa.update_layout(
        xaxis_title="Nível de Interferência (1-5)",
        yaxis_title="Quantidade de Alunos"
    )
    st.plotly_chart(fig_casa, use_container_width=True)

# Terceira linha - Análise de sentimentos
st.subheader("😊 Análise de Sentimentos no Período de Provas")
st.markdown("**Palavras mais usadas para descrever o período de provas:**")

# Análise de palavras (simplificada)
sentimentos = df_filtrado['Resuma em uma palavra como se sente no período de provas.'].dropna()
palavras_contagem = sentimentos.str.lower().str.replace('.', '').value_counts().head(10)

fig_palavras = px.bar(
    x=palavras_contagem.values,
    y=palavras_contagem.index,
    orientation='h',
    color_discrete_sequence=['#96CEB4']
)
fig_palavras.update_layout(
    xaxis_title="Frequência",
    yaxis_title="Sentimento/Palavra"
)
st.plotly_chart(fig_palavras, use_container_width=True)

# Quarta linha - Estratégias de enfrentamento e vícios
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛡️ Estratégias de Enfrentamento Mais Comuns")
    # Análise das estratégias (simplificada)
    estrategias = df_filtrado['Qual estratégia de enfrentamento você usa como estudante?'].value_counts().head(8)
    fig_estrategias = px.bar(
        x=estrategias.values,
        y=estrategias.index,
        orientation='h',
        color_discrete_sequence=['#FECA57']
    )
    fig_estrategias.update_layout(
        xaxis_title="Quantidade",
        yaxis_title="Estratégia"
    )
    st.plotly_chart(fig_estrategias, use_container_width=True)

with col2:
    st.subheader("📊 Acompanhamento Psicológico")
    acompanhamento = df_filtrado['Faz acompanhamento psicológico ou cuidado a saúde mental?'].value_counts()
    fig_acomp = px.pie(
        values=acompanhamento.values,
        names=acompanhamento.index,
        color_discrete_sequence=['#FF9FF3', '#F368E0']
    )
    st.plotly_chart(fig_acomp, use_container_width=True)

# Quinta linha - Dificuldades em trabalhos em grupo
st.subheader("👥 Dificuldades em Trabalhos em Grupo")
dificuldades = df_filtrado['Em poucas palavras quais dificuldades você sente em realizar trabalhos em grupo?'].dropna()

# Mostrar algumas respostas
st.markdown("**Algumas respostas sobre dificuldades em grupo:**")
for i, resposta in enumerate(dificuldades.head(5)):
    st.write(f"{i+1}. {resposta}")

# Seção de dados brutos
st.markdown("---")
st.subheader("📋 Dados Brutos")

with st.expander("Visualizar dados filtrados"):
    st.dataframe(df_filtrado, use_container_width=True)

# Informações do dataset
with st.expander("📊 Estatísticas do Dataset"):
    st.write(f"**Total de registros:** {len(df_filtrado)}")
    st.write(f"**Colunas disponíveis:** {len(df_filtrado.columns)}")
    st.write("**Pré-visualização das colunas:**")
    st.write(list(df_filtrado.columns))

# Footer
st.markdown("---")
st.markdown(
    "**Dashboard desenvolvido para análise de estresse acadêmico** • "
    "Dados coletados via pesquisa com discentes da Univesp"
)