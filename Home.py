import streamlit as st
import pandas as pd

st.set_page_config(
     page_title="Classificação de Notícias em Redes Sociais Sobre a COVID-19 no Brasil",
     page_icon="🌵",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={}
 )


st.header("Classificação de Notícias em Redes Sociais Sobre a COVID-19 no Brasil")

st.subheader("Sobre o projeto")
st.write('Projeto do Trabalho de Conclusão de Curso intitulado "Classificação de Notícias em Redes Sociais sobre a Covid-19 no Brasil", desenvolvido e escrito por Anísio Pereira Batista Filho, discente do discente do curso de Engenharia da Computação da Univasf.')

st.subheader("Resumo")
st.write("A Covid-19, causada pelo coronavírus (Sars-Cov-2), além de seu alto contágio e capacidade de gerar novas variantes, teve um agravante que foi causado pelas desinformações espalhadas nas redes sociais. Assim, identificar e combater tais tipos de informações é de especial importância para o controle da propagação do vírus, visto que a disseminação de notícias falsas tem sido um problema observado em diversos setores da sociedade, dificultando assim, o combate à pandemia. Diante deste contexto, esse trabalho teve como objetivo construir modelos computacionais capazes de classificar, de forma automática, notícias sobre a Covid-19. Para tal, foi ampliada uma base de dados, desenvolvida em um trabalho anterior, para a tarefa de classificação e foi realizada a análise exploratória dos dados obtidos, além do desenvolvimento de modelos de aprendizado supervisionado focados na classificação de notícias sobre o novo coronavírus, considerando as classes notícias verdadeiras (news), falsas (fake) e opiniões (opinion), como resutados, ao comparar os modelos para cda classe, obtivemos que o modelo Random Forest com Oversampling aplicado performou melhor para as classes news com f-score médio igual a 51,2\% e desvio padrão de 2,8\% e fake com f-score médio e desvio padrão de 33,2\% e 5,8\% respectivamente, para a classe opinion o melhor modelo foi o Random Forest sem técincas de balanceamento com 84,3\% de f-score médio e desvio padrão de 1,8\%. Por fim, foi construída uma página web que permite a utilização dos modelos que mais se destacaram para classificar notícias sobre a Covid-19 a partir do uso dos modelos pré-treinados.")

st.subheader("Projeto")
st.write("https://github.com/anisiobfilho/anisiobfilho-tcc-univasf")