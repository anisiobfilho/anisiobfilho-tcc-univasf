import streamlit as st
import pandas as pd

st.set_page_config(
     page_title="Base de Dados",
     page_icon="🌵",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={}
 )

st.experimental_singleton.clear()

## MAIN
st.header("Base de Dados")

@st.experimental_singleton
def carrega_base(path):
    data = pd.read_csv(path, sep=",", low_memory=True)
    data.drop_duplicates(subset='tweet_id', keep='first', inplace=True)
    return data

df = carrega_base('data/corpus_labeled/iguais/bases_tcc/03_geracao_carcteristicas_base.csv')

st.subheader("Covid-19 Text Dataset - CTD")

st.write('Apresentação do corpus gerado no estudo, contendo o ID do tweet e a rotulação correspondente')

st.dataframe(df[['tweet_id', 'label_A']])

st.experimental_singleton.clear()