import streamlit as st
import pandas as pd

st.set_page_config(
     page_title="Projeto Facheiro",
     page_icon="🌵",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={}
 )

st.experimental_singleton.clear()

st.header("Análises Estatísticas 🌵")

@st.experimental_singleton
def carrega_base(path):
    data = pd.read_csv(path, sep=",", low_memory=True)
    return data

st.header('Base de dados estatísticos')
df = carrega_base('models/08_statistics_fscore_base.csv')

st.dataframe(df)

st.header('Comparação entre modelos')
st.write('News')
st.image('models/boxplot/boxplot_news.png')

st.write('Opinion')
st.image('models/boxplot/boxplot_opinion.png')

st.write('Fake')
st.image('models/boxplot/boxplot_fake.png')

st.experimental_singleton.clear()

