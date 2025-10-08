import streamlit as st
import requests
import random
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    """Carrega o modelo de ML treinado a partir do arquivo."""
    try:
        model = joblib.load('modelo_lendario_final.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

def draw_pokemon():
    # O maior ID de Pokémon atualmente é 721
    return random.randint(1, 721)

def get_pokemon_data(pokemon_id):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        nome = data['name']
        imagem = data['sprites']['other']['official-artwork']['front_default']
        return nome, imagem
    else:
        return None, None

def verify_if_is_legendary(pokemon_id):
    url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['is_legendary']
    else:
        return None

@st.cache_resource
def load_artifacts():
    """
    Carrega o modelo de ML treinado e o dataset local a partir dos arquivos.
    """
    try:
        model = joblib.load('models/modelo_lendario_final.joblib')
        
        # Carrega e limpa o DataFrame da mesma forma que no treino
        df = pd.read_csv("raw_data/Pokemon.csv")
        df = df.rename(columns={'#': 'id', "Total": "total"})
        df = df.drop_duplicates(subset=["id"])

        return model, df
    except FileNotFoundError:
        return None, None

def model_predict(ml_model, dataframe, pokemon_id):
    """
    Usa o modelo carregado para prever se um Pokémon é lendário,
    buscando suas features no DataFrame local.
    """
    if ml_model is None or dataframe is None:
        return None
    
    pokemon_features = dataframe[dataframe['id'] == pokemon_id]
    
    if pokemon_features.empty:
        # Se o Pokémon sorteado não estiver no nosso CSV, não conseguimos prever
        return None
    
    # Prepara os dados de entrada no formato que o modelo espera
    # Usando apenas a feature 'total', como no seu script de treino final
    features_para_prever = pokemon_features[['total']]
    
    # Faz a predição
    prediction = ml_model.predict(features_para_prever)
    
    # Retorna o resultado (1 para Lendário, 0 para Comum)
    return bool(prediction[0])

model, df_pokemon_local = load_artifacts()

if model is None or df_pokemon_local is None:
    st.error("Arquivos necessários ('modelo_lendario_final.joblib' ou 'Pokemon.csv') não encontrados!")

st.set_page_config(page_title="Pokémon Lendário?", layout="centered")
st.title("🔮 É um Pokémon Lendário?")

# Estado do app
if "pokemon_id_atual" not in st.session_state:
    st.session_state.pokemon_id_atual = draw_pokemon()
    st.session_state.resposta_usuario = None

# Dados do Pokémon atual
pokemon_id = st.session_state.pokemon_id_atual
nome, imagem_url = get_pokemon_data(pokemon_id)

if nome and imagem_url:
    col1, col2, col3 = st.columns([1, 4, 1]) 
    
    with col1:
        st.write("") 

    with col2:
        # Usamos use_container_width=True para que a imagem preencha a coluna central
        st.image(imagem_url, caption=nome.capitalize(), width=300)

    with col3:
        st.write("") 
else:
    st.error("Erro ao carregar dados do Pokémon.")

# Botões para escolha do usuário
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("⭐ É Lendário"):
        st.session_state.resposta_usuario = True
with col_btn2:
    if st.button("👾 Não é Lendário"):
        st.session_state.resposta_usuario = False

# Mostrar resultados
if st.session_state.resposta_usuario is not None:
    resposta_real = verify_if_is_legendary(pokemon_id)
    resposta_modelo = model_predict(model, df_pokemon_local, pokemon_id)

    st.subheader("🔍 Resultados:")
    st.markdown(f"**Seu palpite:** {'Lendário' if st.session_state.resposta_usuario else 'Não Lendário'}")
    st.markdown(f"**Resposta correta:** {'✅ Lendário' if resposta_real else '❌ Não Lendário'}")
    st.markdown(f"**O modelo acha que é:** {'🧠 Lendário' if resposta_modelo else '🤖 Não Lendário'}")

    if st.button("🔁 Tentar outro Pokémon"):
        st.session_state.pokemon_id_atual = draw_pokemon()
        st.session_state.resposta_usuario = None
        st.rerun() # O novo comando para recarregar a página, mais moderno que experimental_rerun