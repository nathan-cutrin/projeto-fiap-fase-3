import pandas as pd
import joblib
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(filepath="Pokemon.csv"):
    """
    Carrega o dataset a partir de um arquivo CSV.
    """
    return pd.read_csv(filepath)

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações de limpeza e renomeação no DataFrame.
    """
    df_transformed = df.copy()
    df_transformed = df_transformed.drop_duplicates(subset=["#"])
    df_transformed = df_transformed[~df_transformed['Name'].str.contains('Mega')].copy()
    df_transformed['Type 2'].fillna('None', inplace=True)
    df_transformed["Legendary"] = df_transformed["Legendary"].astype(int)
    df_transformed = df_transformed.rename(columns={'#': 'id'})
    df_transformed.columns = (
        df_transformed.columns
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace('sp_', 'special_', regex=False)
    )
    df_transformed.reset_index(drop=True, inplace=True)
    return df_transformed

def generate_and_save_report(model, X_test: pd.DataFrame, y_test: pd.Series, filename="models/classification_report.csv"):
    """
    Gera o classification_report, o converte para um DataFrame e o salva como um arquivo CSV.
    """
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, target_names=['Comum', 'Lendário'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(filename)
    return report_df

def train_final_model_and_save(df: pd.DataFrame, features: list, target: str):
    """
    Treina o modelo final com TODOS os dados e o salva.
    """
    X = df[features]
    y = df[target]

    scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
    
    final_model = LGBMClassifier(
        random_state=42, 
        n_jobs=-1, 
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
    )
    
    final_model.fit(X, y)
    joblib.dump(final_model, 'models/modelo_lendario_final.joblib')

# --- FLUXO DE EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    # 1. Carregar e Transformar
    df_pokemon = load_data("raw_data/Pokemon.csv")
    df_pokemon_clean = transform_data(df_pokemon)
    
    # 2. Definir Features e Alvo
    features_finais = ['total']
    alvo = 'legendary'
    
    X = df_pokemon_clean[features_finais]
    y = df_pokemon_clean[alvo]

    # 3. Dividir os dados para avaliação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 4. Treinar um modelo para avaliação
    scale_pos_weight_eval = (y_train == 0).sum() / (y_train == 1).sum()
    eval_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=scale_pos_weight_eval)
    eval_model.fit(X_train, y_train)
    
    # 5. Gerar e Salvar o Relatório CSV
    report_final = generate_and_save_report(eval_model, X_test, y_test)
    
    # 6. Treinar e Salvar o modelo final com TODOS os dados
    train_final_model_and_save(df_pokemon_clean, features_finais, alvo)