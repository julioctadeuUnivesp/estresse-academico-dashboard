import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
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
            'ansiosa nervosa preocupada estressada sobrecarregada cansada',
            'tranquila calma focada confortável segura',
            'feliz motivada empolgada confiante realizada',
            'triste desanimada desmotivada frustrada',
            'normal regular equilibrada estável'
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

def create_wordcloud(texts, title):
    """Criar nuvem de palavras"""
    all_text = ' '.join([str(text) for text in texts if str(text) != 'Não informado'])
    
    if not all_text.strip():
        return None
        
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
    
    return fig

def main():
    # Carregar dados
    df = load_data()
    
    # Título principal
    st.markdown('<h1 class="main-header">🎓 Dashboard Avançado - Estresse Acadêmico Univesp</h1>', unsafe_allow_html=True)
    
    # Sidebar com filtros avançados
    st.sidebar.header("🔍 Filtros Avançados")
    
    # Filtros múltiplos
    col1, col2, col3, col4 = st.sidebar.columns(4)
    
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
        semestres = ['Todos'] + list(df['Qual semestre se encontra?'].dropna().unique())
        semestre_selecionado = st.selectbox("Semestre", semestres)
    
    # Filtros adicionais na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros Adicionais")
    
    # Filtro por nível de estresse
    estresse_min, estresse_max = st.sidebar.slider(
        "Nível de Estresse nas Provas",
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
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_participantes = len(df_filtrado)
        st.metric("Total de Participantes", total_participantes)
    
    with col2:
        media_estresse = df_filtrado['Em uma escala de 1 a 5, o quanto o período de provas é estressante pra você?'].mean()
        st.metric("Média de Estresse", f"{media_estresse:.1f}/5")
    
    with col3:
        perc_positivo = (df_filtrado['Sentimento_Naive_Bayes'] == 'positivo').mean() * 100
        st.metric("Sentimentos Positivos", f"{perc_positivo:.1f}%")
    
    with col4:
        perc_negativo = (df_filtrado['Sentimento_Naive_Bayes'] == 'negativo').mean() * 100
        st.metric("Sentimentos Negativos", f"{perc_negativo:.1f}%")
    
    with col5:
        perc_acompanhamento = (df_filtrado['Faz acompanhamento psicológico ou cuidado a saúde mental?'] == 'Sim').mean() * 100
        st.metric("Acompanhamento", f"{perc_acompanhamento:.1f}%")
    
    st.markdown("---")
    
    # Primeira linha - Distribuições
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
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Segunda linha - Nuvens de palavras
    st.subheader("☁️ Nuvens de Palavras - Análise Textual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sentimentos no Período de Provas**")
        textos_sentimentos = df_filtrado['Resuma em uma palavra como se sente no período de provas.']
        fig_wc1 = create_wordcloud(textos_sentimentos, "Sentimentos nas Provas")
        if fig_wc1:
            st.pyplot(fig_wc1)
        else:
            st.info("Não há dados textuais para gerar a nuvem de palavras.")
    
    with col2:
        st.markdown("**Dificuldades em Trabalhos em Grupo**")
        textos_dificuldades = df_filtrado['Em poucas palavras quais dificuldades você sente em realizar trabalhos em grupo?']
        fig_wc2 = create_wordcloud(textos_dificuldades, "Dificuldades em Grupo")
        if fig_wc2:
            st.pyplot(fig_wc2)
        else:
            st.info("Não há dados textuais para gerar a nuvem de palavras.")
    
    # Terceira linha - Análise demográfica
    st.subheader("👥 Análise Demográfica")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Distribuição por gênero
        contagem_genero = df_filtrado['Como se identifica?'].value_counts()
        fig_genero = px.bar(
            x=contagem_genero.values,
            y=contagem_genero.index,
            orientation='h',
            title="Distribuição por Gênero",
            color=contagem_genero.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_genero, use_container_width=True)
    
    with col2:
        # Distribuição por raça/etnia
        contagem_raca = df_filtrado['Qual é a sua cor ou raça/etnia?'].value_counts()
        fig_raca = px.pie(
            values=contagem_raca.values,
            names=contagem_raca.index,
            title="Distribuição por Raça/Etnia"
        )
        st.plotly_chart(fig_raca, use_container_width=True)
    
    with col3:
        # Distribuição por eixo do curso
        contagem_curso = df_filtrado['Qual o eixo do seu curso?'].value_counts()
        fig_curso = px.bar(
            x=contagem_curso.index,
            y=contagem_curso.values,
            title="Distribuição por Eixo do Curso",
            color=contagem_curso.values,
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_curso, use_container_width=True)
    
    # Quarta linha - Análise de fatores de estresse
    st.subheader("📊 Fatores de Estresse e Enfrentamento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Interferências nas Rotinas**")
        
        # Criar dataframe para heatmap
        interferencias_data = {
            'Trabalho': df_filtrado['Quanto sua rotina de trabalho interfere nos estudos?'].value_counts().sort_index(),
            'Casa': df_filtrado['Quanto sua rotina de casa interfere nos estudos?'].value_counts().sort_index()
        }
        
        fig_interf = go.Figure()
        
        fig_interf.add_trace(go.Bar(
            name='Interferência do Trabalho',
            x=list(interferencias_data['Trabalho'].index),
            y=interferencias_data['Trabalho'].values,
            marker_color='#3498db'
        ))
        
        fig_interf.add_trace(go.Bar(
            name='Interferência da Casa',
            x=list(interferencias_data['Casa'].index),
            y=interferencias_data['Casa'].values,
            marker_color='#9b59b6'
        ))
        
        fig_interf.update_layout(
            barmode='group',
            xaxis_title="Nível de Interferência (1-5)",
            yaxis_title="Quantidade de Alunos"
        )
        
        st.plotly_chart(fig_interf, use_container_width=True)
    
    with col2:
        st.markdown("**Estratégias de Enfrentamento**")
        
        estrategias = df_filtrado['Qual estratégia de enfrentamento você usa como estudante?'].value_counts().head(10)
        
        fig_estrategias = px.bar(
            x=estrategias.values,
            y=estrategias.index,
            orientation='h',
            title="Estratégias Mais Utilizadas",
            color=estrategias.values,
            color_continuous_scale='thermal'
        )
        
        fig_estrategias.update_layout(
            xaxis_title="Frequência",
            yaxis_title="Estratégia"
        )
        
        st.plotly_chart(fig_estrategias, use_container_width=True)
    
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