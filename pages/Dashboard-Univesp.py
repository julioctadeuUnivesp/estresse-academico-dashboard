import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import warnings
from itertools import cycle
warnings.filterwarnings('ignore')

# Download necessário para o NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Avançado - Estresse Acadêmico",
    page_icon="📚",
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
    df = pd.read_csv('data/academic-stress-level-univesp.csv', sep=',', encoding='utf-8')
        
    # Limpeza básica dos dados
    df.columns = [col.strip().replace('"', '') for col in df.columns]
    df['Qual sua Idade?'] = df['Qual sua Idade?'].fillna(0).astype(int)
    df['Qual semestre se encontra?'] = df['Qual semestre se encontra?'].fillna(0).astype(int)
    df['Atualmente mora só ou divide sua casa?'] = df['Atualmente mora só ou divide sua casa?'].apply(padronizar_residencia)

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
    # Dados de treinamento simples (em produção, usar dataset maior)
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

@st.cache_data
def analyze_sentiment_textblob(text):
    """Análise de sentimentos usando TextBlob (backup)"""
    if pd.isna(text) or text == 'Não informado':
        return 'neutro'
    
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'positivo'
    elif polarity < -0.1:
        return 'negativo'
    else:
        return 'neutro'

def padronizar_residencia(resposta):
    mapeamento_residencias = {
        'Só': 'Sozinho(a)',
        'So': 'Sozinho(a)',
        'Moro': 'Sozinho(a)',
        'só': 'Sozinho(a)',
        'Divido com esposo': 'Família/Conjuge',
        'Moro com esposo': 'Família/Conjuge',
        'com meu esposo': 'Família/Conjuge',
        'Meu marido e eu.': 'Família/Conjuge',
        'Moro com meu esposo': 'Família/Conjuge',
        'Divido com a minha família': 'Família/Conjuge',
        'Moro com a minha família': 'Família/Conjuge',
        'Família': 'Família/Conjuge',
        'Moro com pais idosos': 'Família/Conjuge',
        'casada e com 1 filho': 'Família/Conjuge',
        'Eu e filha.': 'Família/Conjuge',
        'Com minha mãe': 'Família/Conjuge',
        'Moro com marido e 2 filhos': 'Família/Conjuge',
        'Moro com parentes': 'Família/Conjuge',
        'Divido com minha noiva': 'Família/Conjuge',
        'Divido a casa com esposa e um filho.': 'Família/Conjuge',
        'divido com familiares.': 'Família/Conjuge',
        'Divido': 'Outras pessoas',
        'Divido.': 'Outras pessoas',
        'Divido ': 'Outras pessoas',
        'Divido a casa com 6 pessoas': 'Outras pessoas',
        'Divido quintal com minha mãe e irmã': 'Outras pessoas',
        'Divido a casa.': 'Outras pessoas',
        'Divide residência': 'Outras pessoas',
        'Divide': 'Outras pessoas',
        '5': 'Outras pessoas',
        '4': 'Outras pessoas'
    }

    resposta = str(resposta).strip()
    
    # Verificar se está no mapeamento
    if resposta in mapeamento_residencias:
        return mapeamento_residencias[resposta]
    
    # Regras gerais
    if any(termo in resposta.lower() for termo in ['esposo', 'marido', 'filho', 'filha', 'família', 'mãe', 'pai', 'noiva']):
        return 'Família/Conjuge'
    elif any(termo in resposta.lower() for termo in ['divido', 'divide', 'compartilha']):
        return 'Outras pessoas'
    elif any(termo in resposta.lower() for termo in ['só', 'so', 'sozinho', 'moro so']):
        return 'Sozinho(a)'
    else:
        return 'Outros'

def create_plotpie(df, title, color=None):
    """Cria gráfico de pizza"""
    fig = px.pie(
        values=df.values,
        names=df.index,
        title=title,
        color=color
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

def create_boxplot(df, x_col, y_col, title, color_map=None):
    """Cria boxplot para análise de dados categóricos"""
    # Segurança: cópia para não modificar df original
    df = df.copy()
    
    # Verifica se as colunas existem
    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Colunas faltando: {x_col} ou {y_col}")
        return None

    # Preparar coluna x: remover parênteses, preencher nulos e explodir múltiplas opções
    df[x_col] = df[x_col].fillna('').astype(str)
    df[x_col] = df[x_col].str.replace(r'\s*\([^)]*\)', '', regex=True)
    df[x_col] = df[x_col].str.split(r'\s*;\s*')
    df = df.explode(x_col)
    df[x_col] = df[x_col].str.strip()

    # Filtrar valores vazios ou indicadores de "Não informado"
    df = df[~df[x_col].isin(['', 'Não informado', 'nao informado', 'Não informado.'])]

    if df.empty:
        st.warning("Não há dados suficientes para gerar o gráfico.")
        return None

    # Ordem das categorias por frequência
    ordem = df[x_col].value_counts().index.tolist()

    # Gerar mapa de cores consistente se não fornecido
    if not color_map:
        colors = px.colors.qualitative.Plotly
        color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(ordem)}

    # Criar boxplot
    fig_box = px.box(
        df,
        x=x_col,
        y=y_col,
        color=x_col,
        category_orders={x_col: ordem},
        color_discrete_map=color_map,
        title=title,
        height=500
    )

    fig_box.update_xaxes(showticklabels=False)
    return fig_box

# ============================================================================
# COMPONENTES DE UI
# ============================================================================

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

def create_sidebar_filters(df):
    """Cria os filtros na sidebar de forma organizada"""
    st.sidebar.header("🔍 Filtros Avançados")
    
    with st.sidebar.container():
        st.subheader("Filtros Demográficos")
        
        col1, col2 = st.columns(2)
        with col1:
            generos = ['Todos'] + list(df['Como se identifica?'].dropna().unique())
            genero_selecionado = st.selectbox("Gênero", generos)
        
        with col2:
            racas = ['Todos'] + list(df['Qual é a sua cor ou raça/etnia?'].dropna().unique())
            raca_selecionada = st.selectbox("Raça/Etnia", racas)
        
        col3, col4 = st.columns(2)
        with col3:
            eixos = ['Todos'] + list(df['Qual o eixo do seu curso?'].dropna().unique())
            eixo_selecionado = st.selectbox("Eixo do Curso", eixos)
            
        with col4:
            eixos = ['Todos'] + list(df['Atualmente mora só ou divide sua casa?'].dropna().unique())
            convivio_selecionado = st.selectbox("Convívio", eixos)
    
    with st.sidebar.container():
        st.subheader("Filtros Adicionais")
        
        estresse_min, estresse_max = st.slider(
            ":anger: Nível de Estresse (1-5)",
            min_value=1, max_value=5, value=(1, 5),
            help="Filtre pelo nível de estresse relatado"
        )
        
        pressao_social_min, pressao_social_max = st.slider(
            ":busts_in_silhouette: Nível de Pressão Social (1-5)",
            min_value=1, max_value=5, value=(1, 5),
            help="Filtre pelo nível de pressão social relatado"
        )
        
        acompanhamento = st.selectbox(
            "🧠 Acompanhamento Psicológico",
            ['Todos', 'Sim', 'Não'],
            help="Filtre por quem faz acompanhamento psicológico"
        )
    
    with st.sidebar.container():
        st.subheader("🤖 Análise de Sentimentos")
        
        if st.button("Executar Análise Naive Bayes", use_container_width=True):
            st.session_state.run_naive_bayes = True
        else:
            if 'run_naive_bayes' not in st.session_state:
                st.session_state.run_naive_bayes = False
    
    return {
        'genero': genero_selecionado,
        'raca': raca_selecionada,
        'eixo': eixo_selecionado,
        'convivio': convivio_selecionado,
        'estresse_min': estresse_min,
        'estresse_max': estresse_max,
        'pressao_social_min': pressao_social_min,
        'pressao_social_max': pressao_social_max,
        'acompanhamento': acompanhamento
    }

# ============================================================================
# FUNÇÕES DE DISPLAY PRINCIPAIS
# ============================================================================

def apply_filters(df, filters):
    gender = 'Como se identifica?'
    race = 'Qual é a sua cor ou raça/etnia?'
    course_axis = 'Qual o eixo do seu curso?'
    residence = 'Atualmente mora só ou divide sua casa?'
    therapy = 'Faz acompanhamento psicológico ou cuidado a saúde mental?'
    stress_level = 'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'
    social_pressure = 'Sente pressionado por seus colegas de aula ou grupo?'
    
    
    """Aplica os filtros selecionados"""
    df_filtrado = df.copy()
    
    if filters['genero'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado[gender] == filters['genero']]
    
    if filters['raca'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado[race] == filters['raca']]
    
    if filters['eixo'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado[course_axis] == filters['eixo']]
        
    if filters['convivio'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado[residence] == filters['convivio']]
        
    if filters['acompanhamento'] != 'Todos':
        df_filtrado = df_filtrado[df_filtrado[therapy] == filters['acompanhamento']]
    
    # Filtrar por nível de estresse
    df_filtrado = df_filtrado[
        (df_filtrado[stress_level] >= filters['estresse_min']) &
        (df_filtrado[stress_level] <= filters['estresse_max'])
    ]
    
    df_filtrado = df_filtrado[
        (df_filtrado[social_pressure] >= filters['pressao_social_min']) &
        (df_filtrado[social_pressure] <= filters['pressao_social_max'])
    ]

    return df_filtrado

def analyze_sentiments(df_filtrado):
    """Executa análise de sentimentos"""
    textos_sentimentos = df_filtrado['Resuma em uma palavra como se sente no período de provas.']
    
    if st.session_state.run_naive_bayes:
        with st.spinner("Analisando sentimentos com Naive Bayes..."):
            sentimentos = analyze_sentiment_naive_bayes(textos_sentimentos)
    else:
        sentimentos = [analyze_sentiment_textblob(text) for text in textos_sentimentos]
    
    df_filtrado = df_filtrado.copy()
    df_filtrado['Sentimento_Naive_Bayes'] = sentimentos
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
        media_estresse = df_filtrado['Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'].mean()
        create_metric_card(
            f"{media_estresse:.1f}/5", 
            "Stress Provas Média",
            None
        )
    
    with col3:
        perc_positivo = (df_filtrado['Sentimento_Naive_Bayes'] == 'positivo').mean() * 100
        create_metric_card(
            f"{perc_positivo:.1f}%", 
            "Sentimentos Positivos",
            None
        )
    
    with col4:
        perc_negativo = (df_filtrado['Sentimento_Naive_Bayes'] == 'negativo').mean() * 100
        create_metric_card(
            f"{perc_negativo:.1f}%", 
            "Sentimentos Negativos",
            None
        )
    
    # Segunda linha de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        perc_acompanhamento = (df_filtrado['Faz acompanhamento psicológico ou cuidado a saúde mental?'] == 'Sim').mean() * 100
        create_metric_card(
            f"{perc_acompanhamento:.1f}%", 
            "Acompanhamento",
            None
        )
        
    with col6:
        media_ead = df_filtrado['Quanto concorda com a afirmação O estudo online dificulta o aprendizado?'].mean()
        create_metric_card(
            f"{media_ead:.1f}/5", 
            "Dificuldade EAD",
            None
        )
        
    with col7:
        media_presencial = df_filtrado['Dado contexto de ensino a distancia, sente falta de contato presencial com professores e colegas de sala:'].mean()
        create_metric_card(
            f"{media_presencial:.1f}/5", 
            "Preferencia Presencial",
            None
        )
        
    with col8:
        media_pressao = df_filtrado['Sente pressionado por seus colegas de aula ou grupo?'].mean()
        create_metric_card(
            f"{media_pressao:.1f}/5", 
            "Pressão Social",
            None
        )


def display_demographic_analysis(df_filtrado):
    """Exibe análise demográfica"""
    create_section_header("👥 Análise Demográfica")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_plotpie(
            df_filtrado['Como se identifica?'].value_counts(),
            "Distribuição por Gênero"
        )
    
    with col2:
        create_plotpie(
            df_filtrado['Qual é a sua cor ou raça/etnia?'].value_counts(),
            "Distribuição por Raça/Etnia"
        )
    
    with col3:
        create_plotpie(
            df_filtrado['Qual o eixo do seu curso?'].value_counts(),
            "Distribuição por Eixo do Curso"
        )

def display_factors_analysis(df_filtrado):
    """Exibe análise de fatores de influência"""
    create_section_header("🏠 Análise de Fatores de Influência")
    
    tab1, tab2, tab3 = st.tabs(["🏢 Trabalho", "🏠 Casa", ":school: Fatores Acadêmicos"])
    
    with tab1:
        display_work_analysis(df_filtrado)
    
    with tab2:
        display_home_analysis(df_filtrado)
    
    with tab3:
        display_academic_analysis(df_filtrado)

def display_work_analysis(df_filtrado):
    column_model = 'Qual o modelo de trabalho?'
    column_interference = 'Quanto sua rotina de trabalho interfere nos estudos?'
    column_characteristics = 'Marque até 4 características se aplicam ao seu trabalho'
    
    """Análise do ambiente de trabalho"""
    col11, col22 = st.columns(2)
    col12, _  = st.columns(2)
    col21 = st.columns(1)[0]
    
    with col11:
        create_plotpie(
            df_filtrado[column_model].value_counts(),
            "Distribuição por modelo de trabalho",
            None
        )

    with col12:
        fig = px.bar(
            df_filtrado.groupby(column_model)[column_interference]
            .mean()
            .reset_index(),
            x=column_model,
            y=column_interference,
            title='Interferencia do Trabalho',
            labels={
                column_model: 'Modelo de Trabalho',
                column_interference: 'Nível de Estresse'
            },
            color=column_model
        )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        st.plotly_chart(fig, use_container_width=True)

    
    with col21:
        fig = create_boxplot(
            df_filtrado,
            column_characteristics, 
            column_interference, 
            'Interferência do Trabalho nos Estudos'
        )
        if fig:
            fig.update_layout(
                xaxis_title="Características do Trabalho",
                yaxis_title="Interferência nos Estudos",
                legend_title="Características Trabalho"
                )
            st.plotly_chart(fig, use_container_width=True)
    
    with col22:
        # Nuvem de palavras para características do trabalho
        st.subheader("☁️ Características do Ambiente de Trabalho")
        textos_trabalho = df_filtrado[column_characteristics]
        create_wordcloud(textos_trabalho, "Características do Trabalho")


def display_home_analysis(df_filtrado):
    column_residence = 'Atualmente mora só ou divide sua casa?'
    column_interference = 'Quanto sua rotina de casa interfere nos estudos?'
    column_characteristics = 'Marque até 4 características que se aplicam a sua casa.'
    
    """Análise do ambiente doméstico"""
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        create_plotpie(
            df_filtrado[column_residence].value_counts(),
            None
        )

    with col2:
        
        fig = px.bar(
            df_filtrado.groupby(column_residence)[column_interference]
            .mean()
            .reset_index(),
            x=column_residence,
            y=column_interference,
            title='Interferência nos Estudos por Convívio na Residência',
            color=column_residence,
            labels={
            column_residence: 'Convivio na Residência',
            column_interference: 'Interferência Média nos Estudos'
            },
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_boxplot(
            df_filtrado,
            column_characteristics, 
            column_interference, 
            'Interferência da Casa nos Estudos'
        )
        if fig:
            fig.update_layout(
                xaxis_title="Características da Casa",
                yaxis_title="Interferência nos Estudos",
                legend_title="Características Casa"
                )
            st.plotly_chart(fig, use_container_width=True)
        
    with col4:
        st.subheader("☁️ Características do Lar")
        textos_casa = df_filtrado[column_characteristics]
        create_wordcloud(textos_casa, "Características do Lar")


def display_academic_analysis(df_filtrado):
    column_strategy = 'Qual estratégia de enfrentamento você usa como estudante?'
    column_stress = 'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'
    column_pression_social = 'Sente pressionado por seus colegas de aula ou grupo?'
    column_group_difficulty = 'Em poucas palavras quais dificuldades você sente em realizar trabalhos em grupo?'
    
    """Análise de fatores acadêmicos"""
    bar_stress = st.columns(1)[0]
    bar_group = st.columns(1)[0]
    wordCloud, _ = st.columns(2)
    box = st.columns(1)[0]
    
    with bar_stress:
        # Gráfico de barras para ambiente de estudo
        fig = px.bar(
            df_filtrado.groupby(column_strategy)[column_stress]
            .mean()
            .reset_index(),
            x=column_strategy,
            y=column_stress,
            title='Avaliação do Ambiente de Estudo vs Nível de Estresse',
            labels={
                column_strategy: 'Estratégias de Enfrentamento',
                column_stress: 'Nível de Estresse'
            },
            color=column_strategy,
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    with bar_group:
        fig = px.bar(
            df_filtrado.groupby(column_strategy)[column_pression_social]
            .mean()
            .reset_index(),
            x=column_strategy,
            y=column_pression_social,
            title='Estratégias de Enfrentamento vs Nível de Pressão Social',
            labels={
                column_strategy: 'Estratégias de Enfrentamento',
                column_pression_social: 'Nível de Pressão Social'
            },
            color=column_strategy
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    with box:
        # Estratégias de enfrentamento vs estresse
        fig = create_boxplot(
            df_filtrado,
            column_strategy,
            column_stress,
            'Estratégias de Enfrentamento vs Estresse'
        )
        if fig:
            fig.update_layout(
            xaxis_title="Estratégias",
            yaxis_title="Nível de Estresse",
            legend_title="Estratégias"
            )
            
            fig.update_xaxes(showticklabels=False)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with wordCloud:
        # Dificuldades em trabalhos em grupo
        st.subheader("📝 Dificuldades em Trabalhos em Grupo")
        textos_grupo = df_filtrado[column_group_difficulty]
        create_wordcloud(textos_grupo, "Dificuldades em Grupo")


def display_sentiment_behavior_analysis(df_filtrado):
    column_sentiment = 'Sentimento_Naive_Bayes'
    column_feeling = 'Resuma em uma palavra como se sente no período de provas.'
    
    
    """Exibe análise de sentimentos e comportamentos"""
    create_section_header("😰 Análise de Sentimentos e Comportamentos")
    
    tab1, tab2, tab3 = st.tabs(["🎭 Sentimentos", "💪 Hábitos", "🚫 Vícios"])
    
    with tab1:
        display_sentiment_analysis(df_filtrado)
    
    with tab2:
        display_habits_analysis(df_filtrado)
    
    with tab3:
        display_vices_analysis(df_filtrado)

def display_sentiment_analysis(df_filtrado):
    column_sentiment = 'Sentimento_Naive_Bayes'
    column_feeling = 'Resuma em uma palavra como se sente no período de provas.'
    column_stress = 'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'
    
    """Análise de sentimentos"""
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_count = df_filtrado[column_sentiment].value_counts()
        fig_sentimentos = px.pie(
            values=sentiment_count.values,
            names=sentiment_count.index,
            color=sentiment_count.index,
            color_discrete_map={
                'positivo': '#2ecc71',
                'neutro': '#f39c12', 
                'negativo': '#e74c3c'
            },
            title="Distribuição de Sentimentos"
        )
        st.plotly_chart(fig_sentimentos, use_container_width=True)
    
    with col2:
        textos_sentimentos = df_filtrado[column_feeling]
        create_wordcloud(textos_sentimentos, "Sentimentos nas Provas")
    
    # Análise de correlação entre sentimentos e estresse
    st.subheader("📈 Estresse vs Sentimentos")
    fig_box = px.box(
        df_filtrado,
        x=column_sentiment,
        y=column_stress,
        color=column_sentiment,
        color_discrete_map={
            'positivo': '#2ecc71',
            'neutro': '#f39c12',
            'negativo': '#e74c3c'
        }
    )
    st.plotly_chart(fig_box, use_container_width=True)

def display_habits_analysis(df_filtrado):
    column_stress = 'Quanto esse(es) hábito(os) ajuda(am) com seu stress nos estudos?'
    column_habit = 'Marque abaixo até 3 hábitos que tem.'
    
    
    """Análise de hábitos"""
    st.subheader("💪 Hábitos vs Nível de Estresse")
    
    bar_habitos = st.columns(1)[0]
    plot_habitos = st.columns(1)[0]
    
    if not df_filtrado.empty:
        # Preparar dataframe separando múltiplas opções em linhas individuais
        df_habitos = df_filtrado[[column_stress, column_habit]].copy()
        df_habitos[column_habit] = df_habitos[column_habit].fillna('').astype(str)
        
        # Remover parênteses e o conteúdo entre eles
        df_habitos[column_habit] = df_habitos[column_habit].str.replace(r'\s*\([^)]*\)', '', regex=True)
        
        # Dividir por ';' e explodir
        df_habitos[column_habit] = df_habitos[column_habit].str.split(r'\s*;\s*')
        df_habitos = df_habitos.explode(column_habit)
        
        # Limpeza básica dos valores
        df_habitos[column_habit] = df_habitos[column_habit].str.strip()
        df_habitos = df_habitos[df_habitos[column_habit] != '']
        df_habitos = df_habitos[~df_habitos[column_habit].str.lower().isin(['não informado', 'nao informado'])]
    
        if not df_habitos.empty:
            
            with bar_habitos:
                fig_bar = px.bar(
                    df_habitos.groupby(column_habit)[column_stress]
                    .mean()
                    .reset_index(),
                    x=column_habit,  
                    y=column_stress,
                    title='Eficácia Média dos Hábitos no Controle do Estresse',
                    labels={
                        column_habit: 'Tipo de Hábito',
                        column_stress: 'Eficácia Média no Controle do Estresse'
                    },
                    color=column_habit
                )
                fig_bar.update_xaxes(showticklabels=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with plot_habitos:
                fig_box = create_boxplot(
                    df_habitos,
                    column_habit,
                    column_stress,
                    'Eficácia dos Hábitos no Controle do Estresse'
                )
                
                fig_box.update_layout(
                    xaxis_title="Hábitos",
                    yaxis_title="Contribuição no Estresse",
                    showlegend=True,
                    xaxis_tickangle=-45,
                )
                
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.warning("Não há dados suficientes para analisar hábitos.")
    else:
        st.warning("Não há dados filtrados para analisar hábitos.")


def display_vices_analysis(df_filtrado):
    """Análise de vícios"""
    st.subheader("🚫 Vícios vs Contribuição no Estresse")
        
    column_stress = 'O quanto esse(es) vício(os) contribui(em) no seu stress nos estudos?'
    col_vicio = 'Marque abaixo até 3 vícios que tem.'
    
    bar_vicios = st.columns(1)[0]
    plot_vicios = st.columns(1)[0]
    
    if not df_filtrado.empty:
        # Preparar dataframe separando múltiplas opções em linhas individuais
        df_vicios = df_filtrado[[column_stress, col_vicio]].copy()
        df_vicios[col_vicio] = df_vicios[col_vicio].fillna('').astype(str)
        
        # Remover parênteses e o conteúdo entre eles
        df_vicios[col_vicio] = df_vicios[col_vicio].str.replace(r'\s*\([^)]*\)', '', regex=True)
        
        # Dividir por ';' e explodir
        df_vicios[col_vicio] = df_vicios[col_vicio].str.split(r'\s*;\s*')
        df_vicios = df_vicios.explode(col_vicio)
        
        # Limpeza básica dos valores
        df_vicios[col_vicio] = df_vicios[col_vicio].str.strip()
        df_vicios = df_vicios[df_vicios[col_vicio] != '']
        df_vicios = df_vicios[~df_vicios[col_vicio].str.lower().isin(['não informado', 'nao informado'])]
    
        if not df_vicios.empty:
            ordem = df_vicios[col_vicio].value_counts().index.tolist()
            
            with bar_vicios:
                # Gráfico de barras dos vícios
                fig = px.bar(
                    df_vicios.groupby(col_vicio)[column_stress]
                    .mean()
                    .reset_index(),
                    x=col_vicio,  
                    y=column_stress,
                    title='Contribuição Média dos Vícios no Estresse',
                    labels={
                        col_vicio: 'Tipo de Vício',
                        column_stress: 'Contribuição Média no Estresse'
                    },
                    color=col_vicio,
                    category_orders={col_vicio: ordem}
                )      
                
                fig.update_xaxes(showticklabels=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
            
            # Ordenar categorias pelo número de ocorrências
            with plot_vicios:
                # Criar mapa de cores consistente para cada categoria
                colors = px.colors.qualitative.Plotly
                color_cycle = cycle(colors)
                color_map = {cat: next(color_cycle) for cat in ordem}
                
                fig_box = px.box(
                    df_vicios,
                    x=col_vicio,
                    y=column_stress,
                    color=col_vicio,
                    category_orders={col_vicio: ordem},
                    color_discrete_map=color_map,
                    title='Contribuição dos Vícios no Estresse',
                    height=500
                )
                
                fig_box.update_layout(
                    xaxis_title="Tipo de Vício",
                    yaxis_title="Contribuição no Estresse",
                    showlegend=True,
                    xaxis_tickangle=-45,
                )
                
                fig_box.update_xaxes(showticklabels=False)
                
                st.plotly_chart(fig_box, use_container_width=True)
        
        else:
            st.warning("Não há dados filtrados para analisar vícios.")


def display_data_export(df_filtrado):
    """Exibe seção de dados e exportação"""
    create_section_header("📋 Dados e Exportação")
    
    tab1, tab2, tab3 = st.tabs(["📊 Dados Filtrados", "📈 Estatísticas", "💾 Exportar"])
    
    with tab1:
        st.dataframe(df_filtrado, use_container_width=True, height=400)
    
    with tab2:
        display_statistical_analysis(df_filtrado)
    
    with tab3:
        display_export_options(df_filtrado)


def display_statistical_analysis(df_filtrado):
    """Exibe análise estatística"""
    st.subheader("Estatísticas Descritivas")
    
    col1 = st.columns(1)[0]
    
    with col1:
        st.markdown("**Variáveis Numéricas**")
        numeric_cols = df_filtrado.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df_filtrado[numeric_cols].describe())
        else:
            st.info("Não há variáveis numéricas para análise.")

def display_export_options(df_filtrado):
    """Exibe opções de exportação"""
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
        st.info("📊 Dados incluem análise de sentimentos com Naive Bayes")
    
    with col3:
        st.metric("Registros para exportar", len(df_export))

def display_footer():
    """Exibe o footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
        <b>Dashboard desenvolvido com Streamlit</b> • 
        Análise de Estresse Acadêmico • 
        Dados: Pesquisa com Discentes da Univesp
        </div>
        """, 
        unsafe_allow_html=True
    )

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    # Inicializar session state
    if 'run_naive_bayes' not in st.session_state:
        st.session_state.run_naive_bayes = False
    
    # Carregar dados
    df = load_data()
    
    # Título principal
    st.markdown('<h1 class="main-header">🎓 Dashboard de Estresse Acadêmico</h1>', unsafe_allow_html=True)
    
    # Sidebar com filtros
    filters = create_sidebar_filters(df)
    
    # Aplicar filtros
    
    df_filtrado = apply_filters(df, filters)
    
    # Análise de sentimentos
    df_filtrado = analyze_sentiments(df_filtrado)
    
    # Mostrar contador de registros filtrados
    if len(df_filtrado) != len(df):
        st.sidebar.success(f"📈 {len(df_filtrado)} de {len(df)} registros após filtros")
    
    # Seção de métricas
    display_metrics(df_filtrado)
    
    st.markdown("---")
    
    # Análise Demográfica
    display_demographic_analysis(df_filtrado)
    
    st.markdown("---")
    
    # Análise de Fatores
    display_factors_analysis(df_filtrado)
    
    st.markdown("---")
    
    # Análise de Sentimentos e Comportamentos
    display_sentiment_behavior_analysis(df_filtrado)
    
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