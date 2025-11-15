import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

# Download necessário para o NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuração da página
st.set_page_config(
    page_title="Dashboard - Nível de Estresse Acadêmico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv('data/academic-stress-level.csv', sep=',', encoding='utf-8')
    
    # Limpeza básica dos dados e tradução de algumas colunas
    df.columns = [col.strip() for col in df.columns]
    df = df[df['Your Academic Stage'] == 'undergraduate']
    
    # Traduzir colunas
    df.rename(columns={
        'Your Academic Stage': 'Estágio Acadêmico',
        'Peer pressure': 'Pressão dos Colegas',
        'Academic pressure from your home': 'Pressão Familiar',
        'Study Environment': 'Ambiente de Estudo',
        'What coping strategy you use as a student?': 'Estratégia de Enfrentamento',
        'Do you have any bad habits like smoking, drinking on a daily basis?': 'Vícios',
        'What would you rate the academic  competition in your student life': 'Competição Acadêmica',
        'Rate your academic stress index': 'Índice de Estresse'
    }, inplace=True)

    # Traduzir valores
    df['Estágio Acadêmico'] = df['Estágio Acadêmico'].replace({
        'undergraduate': 'Graduação'
        })
    df['Ambiente de Estudo'] = df['Ambiente de Estudo'].replace({
        'Noisy': 'Barulhento',
        'Peaceful': 'Pacífico',
        'disrupted': 'Intermitente'
    })
    df['Vícios'] = df['Vícios'].replace({
        'Yes': 'Sim',
        'No': 'Não',
        'prefer not to say': 'Prefiro não dizer'
    })
    df['Estratégia de Enfrentamento'] = df['Estratégia de Enfrentamento'].replace({
        'Analyze the situation and handle it with intellect':
        'Analisar a situação e lidar com ela com intelecto',
        'Emotional breakdown (crying a lot)':
        'Colapso emocional (chorar muito)',
        'Social support (friends, family)':
        'Suporte social (amigos, família)'
    })

    # Preencher valores vazios
    df.fillna('Não informado', inplace=True)
    
    return df

@st.cache_data
def preprocess_text(text):
    """Pré-processamento de texto para análise de sentimentos"""
    if pd.isna(text) or text == 'Não informado':
        return ''
    
    # Converter para minúsculas e remover caracteres especiais
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text

@st.cache_data
def analyze_sentiment_naive_bayes(texts):
    """Análise de sentimentos usando Naive Bayes"""
    training_data = {
        'texts': [
            'ansiosa nervosa preocupada estressada sobrecarregada cansada triste desanimada desmotivada frustrada exausta',
            'tranquila calma focada confortável segura tranquilo calmo focado seguro empolgado',
            'feliz motivada empolgada confiante realizada bem fácil facil motivado realizado',
            'ansioso nervoso preocupado estressado sobrecarregado cansado desanimado desmotivado frustrado exausto',
            'normal regular equilibrada estável equilibrado'
        ],
        'sentiments': ['negativo', 'neutro', 'positivo', 'negativo', 'neutro']
    }
    
    # Vetorização
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(training_data['texts'])
    y = training_data['sentiments']
    
    # Treinar modelo Naive Bayes
    model = MultinomialNB()
    model.fit(X, y)
    
    # Prever sentimentos
    sentiments = []
    for text in texts:
        processed_text = preprocess_text(text)
        if processed_text:
            X_text = vectorizer.transform([processed_text])
            prediction = model.predict(X_text)[0]
            sentiments.append(prediction)
        else:
            sentiments.append('neutro')
    
    return sentiments

def create_metric_card(value, label, help_text=None):
    """Cria um card de métrica estilizado"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)

def create_section_header(title):
    """Cria um cabeçalho de seção estilizado"""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def create_plotpie(df, title):
    """Cria gráfico de pizza"""
    fig = px.pie(
        values=df.values,
        names=df.index,
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)

def create_wordcloud(texts, title):
    """Criar nuvem de palavras"""
    artigos_preposicoes = set([
        'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas', 'na', 'no',
        'de', 'da', 'do', 'das', 'dos', 'ainda', 'tem', 'sob', 'que', 'q',
        'em', 'para', 'com', 'por', 'sobre', 'ante', 'após', 'apos',
        'entre', 'sem', 'até', 'ate', 'como', 'e', 'ou', 'mas', 'se', 
        'quando', 'onde', 'porque', 'pois', 'como', 'porem', 'porém',
        'contra', 'desde', 'perante', 'trás', 'tras', 'nas', 'nos'
    ])
    
    all_text = ' '.join([str(text) for text in texts if str(text) != 'Não informado'])
    
    if not all_text.strip():
        st.info("Não há dados suficientes para gerar a nuvem de palavras.")
        return
        
    # Remover artigos e preposições
    all_text = ' '.join([word for word in all_text.split() if word not in artigos_preposicoes])
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    
    st.pyplot(fig)

def create_boxplot(df, x_col, y_col, title):
    """Cria boxplot para análise de dados categóricos"""
    df_temp = df.copy()
    
    # Verifica se as colunas existem
    if x_col not in df_temp.columns or y_col not in df_temp.columns:
        st.warning(f"Colunas faltando: {x_col} ou {y_col}")
        return None

    # Filtrar valores vazios
    df_temp = df_temp[df_temp[x_col].notna() & df_temp[y_col].notna()]
    df_temp = df_temp[df_temp[x_col] != 'Não informado']

    if df_temp.empty:
        st.warning("Não há dados suficientes para gerar o gráfico.")
        return None

    # Ordem das categorias por frequência
    ordem = df_temp[x_col].value_counts().index.tolist()

    # Criar boxplot
    fig_box = px.box(
        df_temp,
        x=x_col,
        y=y_col,
        color=x_col,
        category_orders={x_col: ordem},
        title=title,
        height=500
    )

    fig_box.update_xaxes(showticklabels=True)
    return fig_box


def create_bars_graphics(df, x_col, y_col, title, color_col=None, x_label=None, y_label=None):
    fig = px.bar(
        df.groupby(x_col)[y_col]
        .mean()
        .reset_index(),
        x = x_col,
        y = y_col, 
        title = title,
        orientation = 'v',
        color = x_col,
        color_continuous_scale='viridis'
    )

    fig.update_xaxes(showticklabels=False)

    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)
        
    if x_label is not None:
        fig.update_layout(xaxis_title=x_label)

    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)
    

# ============================================================================
# COMPONENTES DE UI
# ============================================================================

def create_sidebar_filters(df):
    """Cria os filtros na sidebar de forma organizada"""
    st.sidebar.header("🔍 Filtros Avançados")
    
    with st.sidebar.container():
        st.subheader("Filtros Demográficos")
        
        # Filtro por ambiente de estudo
        study_envs = ['Todos'] + list(df['Ambiente de Estudo'].dropna().unique())
        study_env_selected = st.selectbox("Ambiente de Estudo", study_envs)
    
    with st.sidebar.container():
        st.subheader("Filtros de Estresse")
        
        # Filtros de pressão
        peer_pressure = st.slider(
            "Pressão dos Colegas (1-5)",
            min_value=1, max_value=5, value=(1, 5)
        )
        
        home_pressure = st.slider(
            "Pressão Familiar (1-5)",
            min_value=1, max_value=5, value=(1, 5)
        )
        
        stress_index = st.slider(
            "Índice de Estresse (1-5)",
            min_value=1, max_value=5, value=(1, 5)
        )
    
    with st.sidebar.container():
        st.subheader("Filtros Comportamentais")
        
        # Filtro por estratégias de coping
        coping_strategies = ['Todos'] + list(df['Estratégia de Enfrentamento'].dropna().unique())
        coping_selected = st.selectbox("Estratégia de Enfrentamento", coping_strategies)
        
        # Filtro por hábitos
        habits = ['Todos'] + list(df['Vícios'].dropna().unique())
        habit_selected = st.selectbox("Vícios", habits)
    
    return {
        'study_env': study_env_selected,
        'peer_pressure_min': peer_pressure[0],
        'peer_pressure_max': peer_pressure[1],
        'home_pressure_min': home_pressure[0],
        'home_pressure_max': home_pressure[1],
        'stress_min': stress_index[0],
        'stress_max': stress_index[1],
        'coping': coping_selected,
        'habit': habit_selected
    }

# ============================================================================
# FUNÇÕES DE DISPLAY PRINCIPAIS
# ============================================================================

def apply_filters(df, filters):
    """Aplica os filtros selecionados"""
    df_filtrado = df.copy()
    
    if filters['study_env'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Ambiente de Estudo'] == filters['study_env']]
    
    if filters['coping'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Estratégia de Enfrentamento'] == filters['coping']]
    
    if filters['habit'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Vícios'] == filters['habit']]
    
    # Filtrar por faixas de valores
    df_filtrado = df_filtrado[
        (df_filtrado['Pressão dos Colegas'] >= filters['peer_pressure_min']) &
        (df_filtrado['Pressão dos Colegas'] <= filters['peer_pressure_max']) &
        (df_filtrado['Pressão Familiar'] >= filters['home_pressure_min']) &
        (df_filtrado['Pressão Familiar'] <= filters['home_pressure_max']) &
        (df_filtrado['Índice de Estresse'] >= filters['stress_min']) &
        (df_filtrado['Índice de Estresse'] <= filters['stress_max'])
    ]
    
    return df_filtrado

def display_metrics(df_filtrado):
    """Exibe as métricas principais de forma organizada"""
    create_section_header("📊 Visão Geral")
    
    # Primeira linha de métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_participantes = len(df_filtrado)
        create_metric_card(
            total_participantes, 
            "Total de Participantes",
            None
        )
    
    with col2:
        media_estresse = df_filtrado['Índice de Estresse'].mean()
        create_metric_card(
            f"{media_estresse:.1f}/5", 
            "Índice de Estresse Médio",
            None
        )
    
    with col3:
        media_pressao_colega = df_filtrado['Pressão dos Colegas'].mean()
        create_metric_card(
            f"{media_pressao_colega:.1f}/5", 
            "Pressão dos Colegas",
            None
        )
    
    with col4:
        media_pressao_familiar = df_filtrado['Pressão Familiar'].mean()
        create_metric_card(
            f"{media_pressao_familiar:.1f}/5", 
            "Pressão Familiar",
            None
        )
    
    # Segunda linha de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        media_competicao = df_filtrado['Competição Acadêmica'].mean()
        create_metric_card(
            f"{media_competicao:.1f}/5", 
            "Competição Acadêmica",
            None
        )
        
    with col6:
        count_bad_habits = len(df_filtrado[df_filtrado['Vícios'] == 'Sim'])
        perc_bad_habits = (count_bad_habits / len(df_filtrado)) * 100 if len(df_filtrado) > 0 else 0
        create_metric_card(
            f"{perc_bad_habits:.1f}%", 
            "Com Vícios",
            None
        )
        
    with col7:
        peaceful_env = len(df_filtrado[df_filtrado['Ambiente de Estudo'] == 'Pacífico'])
        perc_peaceful = (peaceful_env / len(df_filtrado)) * 100 if len(df_filtrado) > 0 else 0
        create_metric_card(
            f"{perc_peaceful:.1f}%", 
            "Ambiente Pacífico",
            None
        )
        
    with col8:
        high_stress = len(df_filtrado[df_filtrado['Índice de Estresse'] >= 4])
        perc_high_stress = (high_stress / len(df_filtrado)) * 100 if len(df_filtrado) > 0 else 0
        create_metric_card(
            f"{perc_high_stress:.1f}%", 
            "Alto Estresse",
            None
        )

def display_demographic_analysis(df_filtrado):
    """Exibe análise demográfica"""
    create_section_header("👥 Análise Demográfica")
    
    col1, col2 = st.columns(2)

    with col1:
        create_plotpie(
            df_filtrado['Ambiente de Estudo'].value_counts(),
            "Distribuição por Ambiente de Estudo"
        )
       
    with col2:
        create_plotpie(
            df_filtrado['Vícios'].value_counts(),
            "Distribuição por Vícios"
        )

def display_pressure_analysis(df_filtrado):
    column_stress = 'Índice de Estresse'
    column_pression_family = 'Pressão Familiar'
    column_pression_friends = 'Pressão dos Colegas'
    column_strategy = 'Estratégia de Enfrentamento'
    
    
    """Exibe análise de pressões"""
    create_section_header("📈 Análise de Pressões e Estresse")
    
    tab1, tab2 = st.tabs(["🎯 Pressões Familiares", "😰 Estratégias de Enfrentamento"])
    
    with tab1:
        col21, col22 = st.columns(2)

        with col21:
            # Pressão dos colegas vs estresse
            fig = create_bars_graphics(
                df_filtrado,
                column_pression_family,
                column_stress,
                'Distribuição',
                None,
                column_pression_family,
                column_stress
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col22:
            # Pressão familiar vs estresse
            fig = create_boxplot(
                df_filtrado,
                'Pressão Familiar',
                'Índice de Estresse',
                'Concentração'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col2 = st.columns(1)[0]
        col1 = st.columns(1)[0] 
        
        with col1:
            # Gráfico de barras das estratégias
        
            create_bars_graphics(
                df_filtrado,
                column_strategy,
                column_stress,
                'Distribuição das Estratégias de Enfrentamento',
                column_strategy,
                column_strategy,
                column_stress
            )
    
        with col2:
            pass

def display_environment_analysis(df_filtrado):
    study_env = 'Ambiente de Estudo'
    competition_acad = 'Competição Acadêmica'
    column_stress = 'Índice de Estresse'
    
    """Exibe análise do ambiente de estudo"""
    create_section_header("🏠 Análise do Ambiente de Estudo")
    
    environment, competition =  st.tabs([":house_with_garden: Ambiente de Estudos", ":men_wrestling: Competição Acadêmica"])

    with environment:    
        col11, col12 = st.columns(2)
        
        with col11:
            
            fig = px.bar(
                    df_filtrado.groupby(study_env)[column_stress]
                    .mean()
                    .reset_index(),
                    x = study_env,
                    y = column_stress, 
                    title = 'Distribuição do Índice de Estresse por Ambiente de Estudo',
                    orientation = 'v',
                    color = study_env,
                    color_continuous_scale='viridis'
                )
            
            fig.update_xaxes(showticklabels=False)
            
            st.plotly_chart(fig, use_container_width=True)

        with col12:
            # Ambiente vs Competição
            fig = create_boxplot(
                df_filtrado,
                'Ambiente de Estudo',
                'Competição Acadêmica',
                'Ambiente de Estudo vs Competição Acadêmica'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
    with competition:   
        col21, col22 = st.columns(2)
     
        with col21:
            fig = px.bar(
                    df_filtrado.groupby(competition_acad)[column_stress]
                    .mean()
                    .reset_index(), 
                    x = competition_acad,
                    y = column_stress, 
                    title = 'Distribuição do Índice de Estresse por Competição Acadêmica',
                    orientation = 'v',
                    color = competition_acad,
                    color_continuous_scale='viridis'
                )
            fig.update_xaxes(showticklabels=False)
            
            st.plotly_chart(fig, use_container_width=True)

        
        with col22:
                # Ambiente vs Estresse
            fig = create_boxplot(
                df_filtrado,
                'Competição Acadêmica',
                'Índice de Estresse',
                'Competição Acadêmica vs Índice de Estresse'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    
    # Estatísticas por ambiente
    st.subheader("📋 Estatísticas por Tipo de Ambiente")
    env_stats = df_filtrado.groupby('Ambiente de Estudo').agg({
        'Índice de Estresse': ['count', 'mean', 'std'],
        'Pressão dos Colegas': 'mean',
        'Pressão Familiar': 'mean'
    }).round(2)
    
    env_stats.columns = ['Contagem', 'Estresse Médio', 'Desvio Padrão', 'Pressão Colegas', 'Pressão Familiar']
    st.dataframe(env_stats, use_container_width=True)

def display_habits_analysis(df_filtrado):
    """Exibe análise de hábitos"""
    create_section_header("🚭 Análise de Hábitos")
    
    plot_vicios = st.columns(1)[0]
    
    with plot_vicios:
        # Hábitos vs Estresse
        fig = create_boxplot(
            df_filtrado,
            'Vícios',
            'Índice de Estresse',
            'Hábitos vs Índice de Estresse'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    


def display_data_export(df_filtrado):
    """Exibe seção de dados e exportação"""
    create_section_header("📋 Dados e Exportação")
    
    tab1, tab2 = st.tabs(["📊 Dados Filtrados", "💾 Exportar"])
    
    with tab1:
        st.dataframe(df_filtrado, use_container_width=True, height=400)
        
        # Estatísticas rápidas
        st.subheader("Estatísticas Rápidas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Variáveis Numéricas**")
            numeric_cols = df_filtrado.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(df_filtrado[numeric_cols].describe())
        
        with col2:
            st.write("**Variáveis Categóricas**")
            categorical_cols = df_filtrado.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"**{col}**: {df_filtrado[col].nunique()} categorias")
    
    with tab2:
        st.subheader("Exportar Dados Analisados")
        
        # Preparar dados para exportação
        df_export = df_filtrado.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Exportar para CSV", use_container_width=True):
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Baixar CSV",
                    data=csv,
                    file_name="dados_estresse_academico_analisado.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.info("📈 Dados incluem todas as análises e filtros aplicados")
        
        with col3:
            st.metric("Registros para exportar", len(df_export))

def display_footer():
    """Exibe o footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
        <b>Dashboard desenvolvido com Streamlit</b> • 
        Análise de Nível de Estresse Acadêmico • 
        Dados: Pesquisa sobre Estresse Acadêmico
        </div>
        """, 
        unsafe_allow_html=True
    )

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    # Inicializar session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Carregar dados
    df = load_data()
    st.session_state.data_loaded = True
    
    # Título principal
    st.markdown('<h1 class="main-header">🎓 Dashboard de Nível de Estresse Acadêmico</h1>', unsafe_allow_html=True)
    
    # Sidebar com filtros
    filters = create_sidebar_filters(df)
    
    # Aplicar filtros
    df_filtrado = apply_filters(df, filters)
    
    # Mostrar contador de registros filtrados
    if len(df_filtrado) != len(df):
        st.sidebar.success(f"📊 {len(df_filtrado)} de {len(df)} registros após filtros")
    else:
        st.sidebar.info(f"📊 Total de {len(df)} registros")
    
    # Seção de métricas
    display_metrics(df_filtrado)
    
    st.markdown("---")
    
    # Análise Demográfica
    display_demographic_analysis(df_filtrado)
    
    st.markdown("---")
    
    # Análise de Pressões
    display_pressure_analysis(df_filtrado)
    
    st.markdown("---")
    
    # Análise do Ambiente
    display_environment_analysis(df_filtrado)
    
    st.markdown("---")
    
    # Análise de Hábitos
    display_habits_analysis(df_filtrado)
    
    st.markdown("---")
    
    # Dados e Exportação
    display_data_export(df_filtrado)
    
    # Footer
    display_footer()

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    main()