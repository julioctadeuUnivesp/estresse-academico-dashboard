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