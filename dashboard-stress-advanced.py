import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #f39c12; }
</style>
""", unsafe_allow_html=True)

# Carregar e preparar dados
@st.cache_data
def load_data():
    df = pd.read_csv('data/estresse_academico_univesp.csv', sep=',', encoding='utf-8')
        
    # Limpeza básica dos dados
    df.columns = [col.strip().replace('"', '') for col in df.columns]
    df['Qual sua Idade?'] = df['Qual sua Idade?'].fillna(0).astype(int)
    df['Qual semestre se encontra?'] = df['Qual semestre se encontra?'].fillna(0).astype(int)
    
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
    'Divido com esposo': 'Divide com família/conjuge',
    'Moro com esposo': 'Divide com família/conjuge',
    'com meu esposo': 'Divide com família/conjuge',
    'Meu marido e eu.': 'Divide com família/conjuge',
    'Moro com meu esposo': 'Divide com família/conjuge',
    'Divido com a minha família': 'Divide com família/conjuge',
    'Moro com a minha família': 'Divide com família/conjuge',
    'Família': 'Divide com família/conjuge',
    'Moro com pais idosos': 'Divide com família/conjuge',
    'casada e com 1 filho': 'Divide com família/conjuge',
    'Eu e filha.': 'Divide com família/conjuge',
    'Com minha mãe': 'Divide com família/conjuge',
    'Moro com marido e 2 filhos': 'Divide com família/conjuge',
    'Moro com parentes': 'Divide com família/conjuge',
    'Divido com minha noiva': 'Divide com família/conjuge',
    'Divido a casa com esposa e um filho.': 'Divide com família/conjuge',
    'divido com familiares.': 'Divide com família/conjuge',
    'Divido': 'Divide com outras pessoas',
    'Divido.': 'Divide com outras pessoas',
    'Divido ': 'Divide com outras pessoas',
    'Divido a casa com 6 pessoas': 'Divide com outras pessoas',
    'Divido quintal com minha mãe e irmã': 'Divide com outras pessoas',
    'Divido a casa.': 'Divide com outras pessoas',
    'Divide residência': 'Divide com outras pessoas',
    'Divide': 'Divide com outras pessoas',
    '5': 'Divide com outras pessoas',
    '4': 'Divide com outras pessoas'}

    resposta = str(resposta).strip()
    
    # Verificar se está no mapeamento
    if resposta in mapeamento_residencias:
        return mapeamento_residencias[resposta]
    
    # Regras gerais
    if any(termo in resposta.lower() for termo in ['esposo', 'marido', 'filho', 'filha', 'família', 'mãe', 'pai', 'noiva']):
        return 'Divide com família/conjuge'
    elif any(termo in resposta.lower() for termo in ['divido', 'divide', 'compartilha']):
        return 'Divide com outras pessoas'
    elif any(termo in resposta.lower() for termo in ['só', 'so', 'sozinho', 'moro so']):
        return 'Sozinho(a)'
    else:
        return 'Outros'



def create_plotpie(df, title):
    dataframe = df
    fig = px.pie(
        values=dataframe.values,
        names=dataframe.index,
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
        return None
        
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
    # Segurança: cópia para não modificar df original
    df = df.copy()
    
    # Verifica se as colunas existem
    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Colunas faltando: {x_col} ou {y_col}")
        return

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
        return

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


def main():
    # Carregar dados
    df = load_data()
    
    # Título principal
    st.markdown('<h1 class="main-header">🎓 Dashboard Estresse Acadêmico Univesp</h1>', unsafe_allow_html=True)
    
    # Sidebar com filtros avançados
    st.sidebar.header("🔍 Filtros Avançados")
    
    # Filtros múltiplos
    col1, col2 = st.sidebar.columns(2)
    col3, col4 = st.sidebar.columns(2)

    with col1:
        # Filtro por gênero
        generos = ['Todos'] + list(df['Como se identifica?'].dropna().unique())
        genero_selecionado = st.selectbox("Gênero", generos)
    
    with col2:
        # Filtro por raça/etnia
        racas = ['Todos'] + list(df['Qual é a sua cor ou raça/etnia?'].dropna().unique())
        raca_selecionada = st.selectbox("Raça/Etnia", racas)
    
    with col3:
        # Filtro por eixo do curso
        eixos = ['Todos'] + list(df['Qual o eixo do seu curso?'].dropna().unique())
        eixo_selecionado = st.selectbox("Eixo do Curso", eixos)
    
    with col4:
        # Filtro por semestre
        semestres = ['Todos'] + list(sorted(df['Qual semestre se encontra?'].dropna().unique()))
        semestre_selecionado = st.selectbox("Semestre", semestres)
    
    # Filtros adicionais na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros Adicionais")
    
    # Filtro por nível de estresse
    estresse_min, estresse_max = st.sidebar.slider(
        "Nível de Estresse (1-5)",
        min_value=1, max_value=5, value=(1, 5)
    )
    
    # Filtro por acompanhamento psicológico
    acompanhamento = st.sidebar.selectbox(
        "Acompanhamento Psicológico",
        ['Todos', 'Sim', 'Não']
    )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    if genero_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Como se identifica?'] == genero_selecionado]
    
    if raca_selecionada != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Qual é a sua cor ou raça/etnia?'] == raca_selecionada]
    
    if eixo_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Qual o eixo do seu curso?'] == eixo_selecionado]
    
    if semestre_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Qual semestre se encontra?'] == semestre_selecionado]
    
    if acompanhamento != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Faz acompanhamento psicológico ou cuidado a saúde mental?'] == acompanhamento]
    
    # Filtrar por nível de estresse
    df_filtrado = df_filtrado[
        (df_filtrado['Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'] >= estresse_min) &
        (df_filtrado['Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'] <= estresse_max)
    ]
    
    # Análise de sentimentos com Naive Bayes
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Análise de Sentimentos")
    
    if st.sidebar.button("Executar Análise Naive Bayes"):
        with st.spinner("Analisando sentimentos com Naive Bayes..."):
            textos_sentimentos = df_filtrado['Resuma em uma palavra como se sente no período de provas.']
            sentimentos = analyze_sentiment_naive_bayes(textos_sentimentos)
            df_filtrado = df_filtrado.copy()
            df_filtrado['Sentimento_Naive_Bayes'] = sentimentos
    else:
        # Usar TextBlob como fallback
        textos_sentimentos = df_filtrado['Resuma em uma palavra como se sente no período de provas.']
        sentimentos = [analyze_sentiment_textblob(text) for text in textos_sentimentos]
        df_filtrado = df_filtrado.copy()
        df_filtrado['Sentimento_Naive_Bayes'] = sentimentos
    
    # Métricas principais
    st.subheader("📊 Métricas Gerais do Dataset Filtrado")
    
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    
    with col1:
        total_participantes = len(df_filtrado)
        st.metric("Total de Participantes", total_participantes)
    
    with col2:
        media_estresse = df_filtrado['Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'].mean()
        st.metric("Stress Provas Média", f"{media_estresse:.1f}/5")
    
    with col3:
        perc_positivo = (df_filtrado['Sentimento_Naive_Bayes'] == 'positivo').mean() * 100
        st.metric("Sentimentos Positivos", f"{perc_positivo:.1f}%")
    
    with col4:
        perc_negativo = (df_filtrado['Sentimento_Naive_Bayes'] == 'negativo').mean() * 100
        st.metric("Sentimentos Negativos", f"{perc_negativo:.1f}%")
    
    with col5:
        perc_acompanhamento = (df_filtrado['Faz acompanhamento psicológico ou cuidado a saúde mental?'] == 'Sim').mean() * 100
        st.metric("Acompanhamento", f"{perc_acompanhamento:.1f}%")
        
    with col6:
        media_ead = df_filtrado['Quanto concorda com a afirmação O estudo online dificulta o aprendizado?'].mean()
        st.metric("Dificuldade EAD", f"{media_ead:.1f}/5")
        
    with col7:
        media_ead = df_filtrado['Dado contexto de ensino a distancia, sente falta de contato presencial com professores e colegas de sala:'].mean()
        st.metric("Preferencia Presencial", f"{media_ead:.1f}/5")
        
    with col8:
        media_ead = df_filtrado['Sente pressionado por seus colegas de aula ou grupo?'].mean()
        st.metric("Pressão Social", f"{media_ead:.1f}/5")
    
    st.markdown("---")
    
    # Primeira linha - Análise demográfica
    st.subheader("👥 Análise Demográfica")
    
    col1, col2 = st.columns(2)

    with col1:
        # Distribuição por gênero
        create_plotpie(
            df_filtrado['Como se identifica?'].value_counts(),
            "Distribuição por Gênero")

    
    with col2:
        # Distribuição por raça/etnia
        create_plotpie(
            df_filtrado['Qual é a sua cor ou raça/etnia?'].value_counts(),
            "Distribuição por Raça/Etnia")

    # Segunda linha - Nuvens de Palavras - Análise Textual
    # st.subheader("☁️ Nuvens de Palavras - Análise Textual")
    
    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader("🎭 Distribuição de Sentimentos (Naive Bayes)")
        sentiment_count = df_filtrado['Sentimento_Naive_Bayes'].value_counts()
        
        fig_sentimentos = px.pie(
            values=sentiment_count.values,
            names=sentiment_count.index,
            color=sentiment_count.index,
            color_discrete_map={
                'positivo': '#2ecc71',
                'neutro': '#f39c12', 
                'negativo': '#e74c3c'
            }
        )
        fig_sentimentos.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sentimentos, use_container_width=True)
    
    with col2:
        st.subheader("📈 Estresse vs Sentimentos")
        
        fig_box = px.box(
            df_filtrado,
            x='Sentimento_Naive_Bayes',
            y='Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?',
            color='Sentimento_Naive_Bayes',
            color_discrete_map={
                'positivo': '#2ecc71',
                'neutro': '#f39c12',
                'negativo': '#e74c3c'
            }
        )
        
        fig_box.update_yaxes(showticklabels=False)
        fig_box.update_xaxes(showticklabels=False)
        st.plotly_chart(fig_box, use_container_width=True)
    

    # Criar abas para diferentes visualizações
    tabWork, tabHome, stress = st.tabs([" 🏢 Trabalho", "🏠 Casa", "😰 Estresse em Período de Provas"])

    with tabWork:
        colCloudWork, colWorkModel = st.columns(2)
        colNivelWork = st.columns(1)[0]

        with colNivelWork:
            fig = create_boxplot(
                        df_filtrado,
                        'Marque até 4 características se aplicam ao seu trabalho', 
                        'Quanto sua rotina de trabalho interfere nos estudos?', 
                        'Interferência nos Estudos vs Características do Trabalho', 
                        {}
                    )
            
            fig.update_layout(
                xaxis_title="Características do Trabalho",
                yaxis_title="Interferência nos Estudos",
                showlegend=True,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)


        with colWorkModel:
            create_plotpie(
                df_filtrado['Qual o modelo de trabalho?'].value_counts(),
                "Modelo de trabalho")


        with colCloudWork:
            st.markdown("**Caracteristicas do Trabalho**")
            
            textos_dificuldades_trabalho = df_filtrado['Marque até 4 características se aplicam ao seu trabalho']
            
            try:
                create_wordcloud(textos_dificuldades_trabalho, "Realidade de Trabalho")
            except:
                st.info("Não há dados textuais para gerar a nuvem de palavras.")
        

    with tabHome:

        colCloudHome, colPieHome = st.columns(2)
        colNivelHome = st.columns(1)[0]

        with colNivelHome:
            fig = create_boxplot(
                df_filtrado,
                'Marque até 4  características que se aplicam a sua casa.', 
                'Quanto sua rotina de casa interfere nos estudos?', 
                'Interferência nos Estudos vs Características do Casa', 
                {}
            )

            fig.update_layout(
                xaxis_title="Características do Casa",
                yaxis_title="Interferência nos Estudos",
                showlegend=True,
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)

        with colPieHome:
            create_plotpie(
                df_filtrado['Atualmente mora só ou divide sua casa?'].apply(padronizar_residencia).value_counts(),
                "Divide moradia"
            )

        with colCloudHome:
            st.markdown("**Caracteristicas do Lar**")
            textos_dificuldades_trabalho = df_filtrado['Marque até 4  características que se aplicam a sua casa.']
            
            try:
                create_wordcloud(textos_dificuldades_trabalho, "Caracteristicas do Lar")
            except:
                st.info("Não há dados textuais para gerar a nuvem de palavras.")
    
    with stress:
        colStressHome = st.columns(1)[0]
        colStressWork = st.columns(1)[0] 

        with colStressHome:
            fig = create_boxplot(
                df_filtrado,
                'Marque até 4 características se aplicam ao seu trabalho', 
                'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?', 
                'Estresse periodo de provas vs Características do Trabalho', 
                {}
            )

            fig.update_layout(
                xaxis_title="Características do Trabalho",
                yaxis_title="Estresse em Período de Provas",
                showlegend=True,
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)

        with colStressWork:
            fig = create_boxplot(
                df_filtrado,
                'Marque até 4  características que se aplicam a sua casa.', 
                'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?', 
                'Estresse periodo de provas vs Características do Casa', 
                {}
            )

            fig.update_layout(
                xaxis_title="Características do Casa",
                yaxis_title="Estresse em Período de Provas",
                showlegend=True,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)





    # Terceira linha - Nuvens de palavras
    st.subheader("Análise Textual")

    tab1, habitos, vicios = st.tabs(["☁️ Nuvens de Palavras", "Habitos", "Vicios"])

    with tab1:
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            pass

        with col2:
            pass
                
        with col3:
            pass
        
        with col4:
            pass
    

    with habitos:
        
        col5 = st.columns(1)[0]
        col6 = st.columns(1)[0]

        with col5:
            st.markdown("**Nível de Estresse vs Hábitos**")
            
            df_habitos = pd.DataFrame()
            color_map = {}  # inicializar para uso posterior em col8
            
            if not df_filtrado.empty:
                col_estresse = 'Quanto esse(es) hábito(os) ajuda(am) com seu stress nos estudos?'
                col_habito = 'Marque abaixo até 3 hábitos que tem.'
            
                # Preparar dataframe separando múltiplas opções em linhas individuais
                df_habitos = df_filtrado[[col_estresse, col_habito]].copy()
                df_habitos[col_habito] = df_habitos[col_habito].fillna('').astype(str)
                
                # Remover parênteses e o conteúdo entre eles (ex: "Café (muito)" -> "Café")
                df_habitos[col_habito] = df_habitos[col_habito].str.replace(r'\s*\([^)]*\)', '', regex=True)
                
                # Dividir por ';' e explodir
                df_habitos[col_habito] = df_habitos[col_habito].str.split(r'\s*;\s*')
                df_habitos = df_habitos.explode(col_habito)
                
                # Limpeza básica dos valores
                df_habitos[col_habito] = df_habitos[col_habito].str.strip()
                df_habitos = df_habitos[df_habitos[col_habito] != '']
                df_habitos = df_habitos[~df_habitos[col_habito].str.lower().isin(['não informado', 'nao informado'])]
            
                if df_habitos.empty:
                    st.warning("Não há dados suficientes para gerar o gráfico.")
                    
                else:
                    # Ordenar categorias pelo número de ocorrências
                    ordem = df_habitos[col_habito].value_counts().index.tolist()
                    
                    # Criar mapa de cores consistente para cada categoria
                    colors = px.colors.qualitative.Plotly
                    color_cycle = cycle(colors)
                    color_map = {cat: next(color_cycle) for cat in ordem}

                    fig_box = create_boxplot(
                        df_habitos,
                        col_habito,
                        col_estresse,
                        'Estresse em Período de Provas x Habito',
                        color_map
                    )
                    
                    fig_box.update_layout(
                        xaxis_title = "Tipo de Habito",
                        yaxis_title = "Estresse em Período de Provas",
                        showlegend = True,
                        xaxis_tickangle=-45,
                    )
                    
                    fig_box.update_xaxes(showticklabels=False)
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                
            else:
                st.warning("Não há dados filtrados para analisar hábitos.")

        
        with col6:
            pass


    with vicios:
                
        col7 = st.columns(1)[0]
        col8 = st.columns(1)[0]

        with col7:
            st.markdown("**Vícios vs. Nível de Estresse**")
            
            df_vicios = pd.DataFrame()
            color_map = {}  # inicializar para uso posterior em col7
            
            if not df_filtrado.empty:
                col_estresse = 'O quanto esse(es) vício(os) contribui(em) no seu stress nos estudos?'
                col_vicio = 'Marque abaixo até 3 vícios que tem.'
            
                # Preparar dataframe separando múltiplas opções em linhas individuais
                df_vicios = df_filtrado[[col_estresse, col_vicio]].copy()
                df_vicios[col_vicio] = df_vicios[col_vicio].fillna('').astype(str)
                
                # Remover parênteses e o conteúdo entre eles (ex: "Café (muito)" -> "Café")
                df_vicios[col_vicio] = df_vicios[col_vicio].str.replace(r'\s*\([^)]*\)', '', regex=True)
                
                # Dividir por ';' e explodir
                df_vicios[col_vicio] = df_vicios[col_vicio].str.split(r'\s*;\s*')
                df_vicios = df_vicios.explode(col_vicio)
                
                # Limpeza básica dos valores
                df_vicios[col_vicio] = df_vicios[col_vicio].str.strip()
                df_vicios = df_vicios[df_vicios[col_vicio] != '']
                df_vicios = df_vicios[~df_vicios[col_vicio].str.lower().isin(['não informado', 'nao informado'])]
            
                if df_vicios.empty:
                    st.warning("Não há dados suficientes para gerar o gráfico.")
                    
                else:
                    # Ordenar categorias pelo número de ocorrências
                    ordem = df_vicios[col_vicio].value_counts().index.tolist()
                    
                    # Criar mapa de cores consistente para cada categoria
                    colors = px.colors.qualitative.Plotly
                    color_cycle = cycle(colors)
                    color_map = {cat: next(color_cycle) for cat in ordem}
                    
                    fig_box = px.box(
                    df_vicios,
                    x=col_vicio,
                    y=col_estresse,
                    color=col_vicio,
                    category_orders={col_vicio: ordem},
                    color_discrete_map=color_map,
                    title='Estresse em Período de Provas x Vício',
                    height=500
                    )
                    
                    fig_box.update_layout(
                    xaxis_title = "Tipo de Vício",
                    yaxis_title = "Estresse em Período de Provas",
                    showlegend = True,
                    xaxis_tickangle=-45,
                    )
                    
                    fig_box.update_xaxes(showticklabels=False)
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                
            else:
                st.warning("Não há dados filtrados para analisar vícios.")        
        
        with col8:
            pass


    # Quarta linha - Análise de fatores de estresse
    st.subheader("📊 Fatores de Estresse e Enfrentamento")
    
    # Criar abas para diferentes visualizações
    tab1, tab2 = st.tabs(["🎯 Estratégias vs Estresse", "📋 Estatísticas"])
    
    with tab1:
        st.markdown("**Estratégias de Enfrentamento vs. Nível de Estresse**")
        
        if not df_filtrado.empty:
            # Boxplot principal
            fig = create_boxplot(
                df_filtrado,
                'Qual estratégia de enfrentamento você usa como estudante?',
                'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?',
                'Distribuição do Nível de Estresse por Estratégia',
                {})
            
            fig.update_layout(
                xaxis_title="Estratégia de Enfrentamento",
                yaxis_title="Estresse em Período de Provas",
                showlegend=True,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Não há dados suficientes para gerar o gráfico.")
    
    with tab2:
        st.markdown("**Estatísticas Detalhadas por Estratégia**")
        
        if not df_filtrado.empty:
            # Tabela de estatísticas
            stats_estrategias = df_filtrado.groupby('Qual estratégia de enfrentamento você usa como estudante?').agg({
                'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?': ['count', 'mean', 'std', 'min', 'max'],
                'Quanto sua rotina de trabalho interfere nos estudos?': 'mean',
                'Quanto sua rotina de casa interfere nos estudos?': 'mean'
            }).round(2)
            
            # Renomear colunas para melhor visualização
            stats_estrategias.columns = ['Contagem', 'Média Estresse', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Interf. Trabalho', 'Interf. Casa']
            
            st.dataframe(stats_estrategias, use_container_width=True)
            
            # Gráfico de barras com médias
            fig_barras = px.bar(
                stats_estrategias.reset_index(),
                x='Qual estratégia de enfrentamento você usa como estudante?',
                y='Média Estresse',
                color='Média Estresse',
                color_continuous_scale='viridis',
                title='Média de Estresse por Estratégia de Enfrentamento'
            )
            
            fig_barras.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_barras, use_container_width=True)
    


    # Quinta linha - Análise detalhada dos sentimentos
    st.subheader("🔍 Análise Detalhada dos Sentimentos")
    
    # Tabela com exemplos de sentimentos
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Exemplos de Respostas por Sentimento**")
        
        sentiment_groups = df_filtrado.groupby('Sentimento_Naive_Bayes')
        
        for sentimento, group in sentiment_groups:
            st.markdown(f"**{sentimento.upper()}**")
            exemplos = group['Resuma em uma palavra como se sente no período de provas.'].head(3).tolist()
            for exemplo in exemplos:
                if exemplo != 'Não informado':
                    st.write(f"• {exemplo}")
            st.write("")
    
    with col2:
        st.markdown("**Estatísticas por Sentimento**")
        
        stats_sentimento = df_filtrado.groupby('Sentimento_Naive_Bayes').agg({
            'Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?': 'mean',
            'Quanto sua rotina de trabalho interfere nos estudos?': 'mean',
            'Quanto sua rotina de casa interfere nos estudos?': 'mean'
        }).round(2)
        
        st.dataframe(stats_sentimento)


    # Criar abas para diferentes visualizações
    provas, grupos = st.tabs(["📈 Provas", "🎯 Grupos"])
    
    with provas:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sentimentos no Período de Provas**")
            textos_sentimentos = df_filtrado['Resuma em uma palavra como se sente no período de provas.']
            try:
                create_wordcloud(textos_sentimentos, "Sentimentos nas Provas")
            except:
                st.info("Não há dados textuais para gerar a nuvem de palavras.")
        
        with col2:
            pass


    with grupos:
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Dificuldades em Trabalhos em Grupo**")
            textos_dificuldades = df_filtrado['Em poucas palavras quais dificuldades você sente em realizar trabalhos em grupo?']
            
            try:
                create_wordcloud(textos_dificuldades, "Dificuldades em Grupo")
            except:
                st.info("Não há dados textuais para gerar a nuvem de palavras.")

        with col4:
            pass




    # Seção de dados brutos com análise
    st.markdown("---")
    st.subheader("📋 Dados Detalhados e Exportação")
    
    tab1, tab2, tab3 = st.tabs(["Dados Filtrados", "Análise Estatística", "Exportar Dados"])
    
    with tab1:
        st.dataframe(df_filtrado, use_container_width=True)
    
    with tab2:
        st.subheader("Estatísticas Descritivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Variáveis Numéricas**")
            numeric_cols = df_filtrado.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(df_filtrado[numeric_cols].describe())
            else:
                st.info("Não há variáveis numéricas para análise.")
        
        with col2:
            st.markdown("**Correlações**")
            if len(numeric_cols) > 1:
                corr_matrix = df_filtrado[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("Exportar Dados Analisados")
        
        # Preparar dados para exportação
        df_export = df_filtrado.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Exportar para CSV"):
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Baixar CSV",
                    data=csv,
                    file_name="dados_estresse_academico_analisado.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.info("Dados incluem análise de sentimentos com Naive Bayes")
        
        with col3:
            st.metric("Registros para exportar", len(df_export))
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <b>Dashboard desenvolvido com Streamlit</b> • 
        Análise de Estresse Acadêmico com Naive Bayes • 
        Dados: Pesquisa com Discentes da Univesp
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()