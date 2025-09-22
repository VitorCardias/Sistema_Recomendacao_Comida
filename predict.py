import pandas as pd
import numpy as np
import joblib
import ast
from sklearn.feature_extraction.text import CountVectorizer

# ==============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO (copiadas do notebook de treinamento)
# ==============================================================================

def tratar_coluna_nutricao(df):
    """Transforma a coluna 'nutrition' em colunas numéricas separadas."""
    df_copy = df.copy()
    coluna_nutrition_str = df_copy['nutrition'].fillna('[]').astype(str)
    try:
        lista_nutrition = coluna_nutrition_str.apply(ast.literal_eval)
        if lista_nutrition.apply(len).nunique() > 1:
            print("Aviso: As listas de nutrição têm comprimentos diferentes. Padronizando...")
            target_len = lista_nutrition.apply(len).max()
            lista_nutrition = lista_nutrition.apply(lambda x: x + [0] * (target_len - len(x)) if len(x) < target_len else x[:target_len])

        df_nutrition = pd.DataFrame(lista_nutrition.tolist(), index=df_copy.index)
        df_nutrition.columns = [f'nutri_{i}' for i in range(df_nutrition.shape[1])]
        df_final = pd.concat([df_copy, df_nutrition], axis=1)
        df_final = df_final.drop('nutrition', axis=1)
        return df_final
    except Exception as e:
        print(f"Erro ao processar a coluna de nutrição: {e}")
        return df_copy.drop('nutrition', axis=1)


def criar_features_receitas(df_receitas, df_train, vectorizer):
    """Cria todas as features relacionadas às receitas."""
    media_rating_por_item = df_train.groupby('recipe_id')['rating'].mean().rename('media_rating_item')
    num_avaliacoes_por_item = df_train.groupby('recipe_id')['rating'].count().rename('num_avaliacoes_item')
    df_receitas = df_receitas.merge(media_rating_por_item, left_on='id', right_index=True, how='left')
    df_receitas = df_receitas.merge(num_avaliacoes_por_item, left_on='id', right_index=True, how='left')

    df_receitas['submitted'] = pd.to_datetime(df_receitas['submitted'])
    df_receitas['ano_publicacao'] = df_receitas['submitted'].dt.year
    df_receitas['mes_publicacao'] = df_receitas['submitted'].dt.month
    data_recente = df_receitas['submitted'].max()
    df_receitas['idade_receita_dias'] = (data_recente - df_receitas['submitted']).dt.days

    df_receitas['tags_limpas'] = df_receitas['tags'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    tags_matrix = vectorizer.transform(df_receitas['tags_limpas'])
    df_tags = pd.DataFrame(tags_matrix.toarray(), index=df_receitas.index, columns=['tag_' + tag for tag in vectorizer.get_feature_names_out()])
    df_receitas = pd.concat([df_receitas, df_tags], axis=1)

    df_receitas = tratar_coluna_nutricao(df_receitas)

    media_geral_rating = df_train['rating'].mean()
    df_receitas['item_desvio_media_geral'] = df_receitas['media_rating_item'] - media_geral_rating

    return df_receitas, media_geral_rating


def criar_features_usuarios(df_train):
    """Cria features relacionadas aos usuários (com base nos dados de TREINO)."""
    media_rating_por_usuario = df_train.groupby('user_id')['rating'].mean().rename('media_rating_usuario')
    num_avaliacoes_por_usuario = df_train.groupby('user_id')['rating'].count().rename('num_avaliacoes_usuario')
    df_users = pd.DataFrame(media_rating_por_usuario).join(num_avaliacoes_por_usuario)
    return df_users


# ==============================================================================
# FUNÇÃO PRINCIPAL DE PREVISÃO
# ==============================================================================

def predict(data_path, models_path):
    print("--- Iniciando o processo de previsão ---")

    print("Carregando modelo e vectorizer...")
    try:
        model = joblib.load(f"{models_path}/modelo_xgboost_final.joblib")
        vectorizer = joblib.load(f"{models_path}/tags_vectorizer.joblib")
    except FileNotFoundError:
        print("Erro: Arquivos de modelo (.joblib) não encontrados. Verifique o caminho.")
        return

    print("Carregando dados brutos...")
    try:
        df_receitas_raw = pd.read_csv(f"{data_path}/RAW_recipes.csv")
        df_train_raw = pd.read_csv(f"{data_path}/interactions_train.csv")
        df_to_predict = pd.read_csv(f"{data_path}/interactions_test.csv")
    except FileNotFoundError:
        print("Erro: Arquivos de dados (.csv) não encontrados. Verifique o caminho.")
        return

    print("Recriando features de receitas e usuários...")
    df_receitas_featured, media_global = criar_features_receitas(df_receitas_raw, df_train_raw, vectorizer)
    df_users_featured = criar_features_usuarios(df_train_raw)

    print("Montando o conjunto de dados para previsão...")
    df_final = df_to_predict.merge(df_receitas_featured, left_on='recipe_id', right_on='id', how='left')
    df_final = df_final.merge(df_users_featured, on='user_id', how='left')

    # CORREÇÃO PANDAS: Substituindo inplace=True para evitar o FutureWarning
    df_final['media_rating_item'] = df_final['media_rating_item'].fillna(media_global)
    df_final['media_rating_usuario'] = df_final['media_rating_usuario'].fillna(media_global)
    df_final.fillna(0, inplace=True)

    # ==============================================================================
    # ADIÇÃO DA FEATURE FALTANTE - A CAUSA DO ERRO
    # ==============================================================================
    print("Criando a feature 'rating_desvio_usuario'...")
    df_final['rating_desvio_usuario'] = df_final['rating'] - df_final['media_rating_usuario']
    # ==============================================================================

    print("Selecionando features para o modelo...")
    features_usadas_no_treino = model.get_booster().feature_names
    
    # Checagem de segurança para garantir que todas as colunas existem
    for col in features_usadas_no_treino:
        if col not in df_final.columns:
            print(f"ERRO CRÍTICO: A coluna esperada pelo modelo '{col}' não foi encontrada após o processamento.")
            return

    X_predict = df_final[features_usadas_no_treino]

    print("Realizando as previsões...")
    predictions = model.predict(X_predict)

    print("Previsões concluídas!")
    df_resultado = df_to_predict.copy()
    df_resultado['predicted_rating'] = predictions
    df_resultado['predicted_rating'] = np.round(df_resultado['predicted_rating'], 2)

    print("\n--- Amostra dos Resultados ---")
    print(df_resultado.head(10))

    output_path = "predictions.csv"
    df_resultado.to_csv(output_path, index=False)
    print(f"\nResultados salvos com sucesso em: {output_path}")


if __name__ == "__main__":
    DATA_PATH = 'archive'
    MODELS_PATH = 'models'
    predict(DATA_PATH, MODELS_PATH)