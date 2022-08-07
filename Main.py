import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

## Abertura de arquivo e criação do dataframe:
df = pd.read_csv('data/corpus_labeled/iguais/bases_tcc/03_geracao_carcteristicas_base.csv', sep=",", low_memory=False)
dfA1 = pd.read_csv("data/corpus_labeled/rotulaçao1[anisiofilho].csv", sep=",", low_memory=True)
dfA1.drop_duplicates(subset='tweet_id', keep='first', inplace=True)
dfB1 = pd.read_csv("data/corpus_labeled/rotulaçao1[debora].csv", sep=",", low_memory=True)
dfB1.drop_duplicates(subset='tweet_id', keep='first', inplace=True)
dfA2 = pd.read_csv("data/corpus_labeled/rotulaçao2[anisiofilho].csv", sep=",", low_memory=True)
dfA2.drop_duplicates(subset='tweet_id', keep='first', inplace=True)
dfB2 = pd.read_csv("data/corpus_labeled/rotulaçao2[andre].csv", sep=",", low_memory=True)
dfB2.drop_duplicates(subset='tweet_id', keep='first', inplace=True)


## Funções
### Módulos

#### Ajustador de turnos
def ajusta_turno(linha):
    return linha.capitalize()

##### Ajustador de usuários
def ajusta_usuarios(linha):
    if linha == 'invalid_user':
        return 'Usuário Inválido'
    else:
        return linha

#### Ajustador de estados
def ajusta_estados(linha):
    if linha == 'invalidstate':
        return 'Inválido (Usuário não informou sua localização)'
    elif linha == 'notbrazilstate':
        return 'Exterior (Fora do Brasil)'
    elif linha == 'statenotdefined':
        return 'Brasil (Localização definida apenas como Brasil)'
    else:
        return linha

#### Ajustador de regiões
def ajusta_regioes(linha):
    if linha == 'invalidregion':
        return 'Inválido<br>(Usuário não informou sua localização)'
    elif linha == 'notbrazilregion':
        return 'Exterior<br>(Fora do Brasil)'
    elif linha == 'regionnotdefined':
        return 'Brasil<br>(Localização definida apenas como Brasil)'
    else:
        return linha

#### Ajustador de rótulos
def ajusta_rotulo(linha):
    if linha == -1:
        return 'Notícia Verdadeira'
    elif linha == 0:
        return 'Opinião'
    elif linha == 1:
        return 'Notícia Falsa'

### Contador de frequência de palavras:
def gera_frequencia_palavras(dicionario_frequencia, dataframe, coluna, rotulo):
    for index, row in dataframe.iterrows():
        if row.label_A == rotulo:
            row[coluna] = eval(row[coluna])
            for palavra in row[coluna]:
                
                if palavra not in dicionario_frequencia.keys():
                    dicionario_frequencia[palavra] = 1
                else:
                    dicionario_frequencia[palavra] += 1
    
    dicionario_frequencia = dict(sorted(dicionario_frequencia.items(), key=lambda item: item[1]))
    
    return dicionario_frequencia

### Gerador de nuvem de palavras
def gera_nuvem_palavras(dicionario_frequencia):
    wordcloud = WordCloud(#font_path=None, 
                        width=800, 
                        height=400, 
                        margin=2, 
                        ranks_only=None, 
                        prefer_horizontal=0.9, 
                        mask=None, 
                        scale=1, 
                        color_func=None, 
                        max_words=500, 
                        min_font_size=4, 
                        stopwords=None, 
                        random_state=None, 
                        background_color='white', 
                        max_font_size=None, 
                        font_step=1, 
                        mode='RGB', 
                        relative_scaling='auto', 
                        regexp=None, 
                        collocations=True, 
                        colormap='hsv', 
                        normalize_plurals=True, 
                        contour_width=0, 
                        contour_color='black', 
                        repeat=False, 
                        include_numbers=False, 
                        min_word_length=0, 
                        collocation_threshold=30).generate_from_frequencies(dicionario_frequencia)
    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.imshow(wordcloud, interpolation='bilinear')       
    ax.set_axis_off()
    
    return fig, wordcloud

## Main
### Análise exploratória da base de dados
#### Gráfico de turnos
df_turnos_grafico = df.time_shift.value_counts().reset_index()
df_turnos_grafico.rename(columns={'index':'Turno','time_shift':'Quantidade'}, inplace=True)
df_turnos_grafico['Turno'] = df_turnos_grafico['Turno'].map(ajusta_turno)

fig1 = px.bar(df_turnos_grafico, 
             x='Turno', 
             y='Quantidade', 
             title='Quantidade de postagens por turno', 
             text_auto=True,
             width=1000,
             height=600,
            )
st.plotly_chart(fig1, use_container_width=True)

#### Gráfico de nomes de usuários
df_usuarios_grafico = df.user_screen_name.value_counts().reset_index()[0:10]
df_usuarios_grafico.rename(columns={'index':'Nome de usuário','user_screen_name':'Quantidade'}, inplace=True)
df_usuarios_grafico.sort_values(by='Quantidade', ascending=True, inplace=True)
df_usuarios_grafico['Nome de usuário'] = df_usuarios_grafico['Nome de usuário'].map(ajusta_usuarios)

fig2 = px.bar(df_usuarios_grafico, 
             y='Nome de usuário', 
             x='Quantidade', 
             title='Top 10: Usuários que mais postam', 
             text_auto=True,
             orientation='h',
             width=1000,
             height=600,
            )
st.plotly_chart(fig2, use_container_width=True)

#### Gráfico de Estados
df_estados_grafico = df.state_location.value_counts().reset_index()
df_estados_grafico.rename(columns={'index':'Estado','state_location':'Quantidade'}, inplace=True)
df_estados_grafico.sort_values(by='Quantidade', ascending=True, inplace=True)
df_estados_grafico['Estado'] = df_estados_grafico['Estado'].map(ajusta_estados)

fig3 = px.bar(df_estados_grafico, 
             y='Estado', 
             x='Quantidade', 
             title='Quantidade de postagens por Unidade Federativa', 
             text_auto=True, 
             orientation='h',
             width=1000,
             height=600,
            )
st.plotly_chart(fig3, use_container_width=True)

#### Gráfico de Regiões
df_regioes_grafico = df.region_location.value_counts().reset_index()
df_regioes_grafico.rename(columns={'index':'Região','region_location':'Quantidade'}, inplace=True)
df_regioes_grafico.sort_values(by='Quantidade', ascending=True, inplace=True)
df_regioes_grafico['Região'] = df_regioes_grafico['Região'].map(ajusta_regioes)

fig4 = px.bar(df_regioes_grafico, 
             y='Região', 
             x='Quantidade', 
             title='Quantidade de postagens por Região do Brasil', 
             text_auto=True, 
             orientation='h',
             width=1000,
             height=600,
            )
st.plotly_chart(fig4, use_container_width=True)

#### Gráfico de rótulo por turno
df_turnos_rotulos = df[['time_shift', 'label_A']].value_counts().reset_index()
df_turnos_rotulos.rename(columns={'time_shift':'Turno','label_A':'Rótulo', 0:'Quantidade'}, inplace=True)
df_turnos_rotulos.sort_values(by='Quantidade', ascending=True, inplace=True)
df_turnos_rotulos['Turno'] = df_turnos_rotulos['Turno'].map(ajusta_turno) 
df_turnos_rotulos['Rótulo'] = df_turnos_rotulos['Rótulo'].map(ajusta_rotulo)

fig5 = px.bar(df_turnos_rotulos, 
             x='Rótulo', 
             y='Quantidade', 
             color='Turno',
             color_discrete_map={
                                    'Manhã':'tomato',
                                    'Tarde':'yellowgreen',
                                    'Noite':'steelblue',
                                },
             barmode='group',
             title='Quantidade de postagens de cada rótulo por turno',
             text_auto=True,
             width=1000,
             height=600,
            )
st.plotly_chart(fig5, use_container_width=True)

#### Gráfico de rótulo por região
df_regioes_rotulos = df[['region_location', 'label_A']].value_counts().reset_index()
df_regioes_rotulos.rename(columns={'region_location':'Região','label_A':'Rótulo', 0:'Quantidade'}, inplace=True)
df_regioes_rotulos.sort_values(by='Quantidade', ascending=True, inplace=True)
df_regioes_rotulos['Rótulo'] = df_regioes_rotulos['Rótulo'].map(ajusta_rotulo) 
df_regioes_rotulos['Região'] = df_regioes_rotulos['Região'].map(ajusta_regioes)

fig6 = px.bar(df_regioes_rotulos, 
             y='Região', 
             x='Quantidade', 
             color='Rótulo',
             color_discrete_map={
                                    'Notícia Falsa':'tomato',
                                    'Notícia Verdadeira':'yellowgreen',
                                    'Opinião':'steelblue',
                                },
             barmode='group',
             title='Quantidade de postagens de cada rótulo por turno', 
             text_auto=True,
             orientation='h',
             width=1000,
             height=600,
            )
st.plotly_chart(fig6, use_container_width=True)

#### Gráfico de rótulo por nome de usuário
lista_usuarios_top = df_usuarios_grafico['Nome de usuário'].to_list()
lista_usuarios_top.append('invalid_user')

df_usuario_rotulos = df[['user_screen_name', 'label_A']].value_counts().reset_index()
df_usuario_rotulos = df_usuario_rotulos[df_usuario_rotulos['user_screen_name'].isin(lista_usuarios_top)]
df_usuario_rotulos.rename(columns={'user_screen_name':'Nome de usuário','label_A':'Rótulo', 0:'Quantidade'}, inplace=True)
df_usuario_rotulos.sort_values(by=['Quantidade'], ascending=True, inplace=True)
df_usuario_rotulos['Rótulo'] = df_usuario_rotulos['Rótulo'].map(ajusta_rotulo)
df_usuario_rotulos['Nome de usuário'] = df_usuario_rotulos['Nome de usuário'].map(ajusta_usuarios)

fig7 = px.bar(df_usuario_rotulos, 
             y='Nome de usuário', 
             x='Quantidade', 
             color='Rótulo',
             color_discrete_map={
                                    'Notícia Falsa':'tomato',
                                    'Notícia Verdadeira':'yellowgreen',
                                    'Opinião':'steelblue',
                                },
             barmode='group',
             title='Quantidade de postagens de cada rótulo por nome de usuário', 
             text_auto=True,
             orientation='h',
             width=1000,
             height=600,
            )
st.plotly_chart(fig7, use_container_width=True)

### Análises por classe entre os rotuladores
#### Gráfico de distribuição de classes entre os rotuladores A1 e B1
df_dist_rotuloA1 = dfA1['label'].value_counts().reset_index()
df_dist_rotuloA1['Rotulador'] = 'A1'
df_dist_rotuloA1.rename(columns={'index':'Rótulo', 'label':'Quantidade'}, inplace=True)
df_dist_rotuloA1['Rótulo'] = df_dist_rotuloA1['Rótulo'].map(ajusta_rotulo)

df_dist_rotuloB1 = dfB1['label'].value_counts().reset_index()
df_dist_rotuloB1['Rotulador'] = 'B1'
df_dist_rotuloB1.rename(columns={'index':'Rótulo', 'label':'Quantidade'}, inplace=True)
df_dist_rotuloB1['Rótulo'] = df_dist_rotuloB1['Rótulo'].map(ajusta_rotulo)

df_dist_rotuloA1B1 = pd.concat([df_dist_rotuloA1, df_dist_rotuloB1])

fig8 = px.bar(df_dist_rotuloA1B1, 
             x='Rótulo', 
             y='Quantidade', 
             color='Rotulador',
             color_discrete_map={
                                    #'A1':'tomato',
                                    'A1':'yellowgreen',
                                    'B1':'steelblue',
                                },
             barmode='group',
             title='Quantidade de cada rótulo por rotulador', 
             text_auto=True,
             width=1000,
             height=600,
            )
st.plotly_chart(fig8, use_container_width=True)


#### Gráfico de distribuição de classes entre os rotuladores A2 e B2
df_dist_rotuloA2 = dfA2['label'].value_counts().reset_index()
df_dist_rotuloA2['Rotulador'] = 'A2'
df_dist_rotuloA2.rename(columns={'index':'Rótulo', 'label':'Quantidade'}, inplace=True)
df_dist_rotuloA2['Rótulo'] = df_dist_rotuloA2['Rótulo'].map(ajusta_rotulo)

df_dist_rotuloB2 = dfB2['label'].value_counts().reset_index()
df_dist_rotuloB2['Rotulador'] = 'B2'
df_dist_rotuloB2.rename(columns={'index':'Rótulo', 'label':'Quantidade'}, inplace=True)
df_dist_rotuloB2['Rótulo'] = df_dist_rotuloB2['Rótulo'].map(ajusta_rotulo)

df_dist_rotuloA2B2 = pd.concat([df_dist_rotuloA2, df_dist_rotuloB2])

fig9 = px.bar(df_dist_rotuloA2B2, 
             x='Rótulo', 
             y='Quantidade', 
             color='Rotulador',
             color_discrete_map={
                                    #'A1':'tomato',
                                    'A2':'yellowgreen',
                                    'B2':'steelblue',
                                },
             barmode='group',
             title='Quantidade de cada rótulo por rotulador', 
             text_auto=True,
             width=1000,
             height=600,
            )
st.plotly_chart(fig9, use_container_width=True)

### Criação de wordclouds por rótulo
#### Tweet_text_stemming
##### Fake News
stemming_fake_dict = dict()
stemming_fake_dict = gera_frequencia_palavras(stemming_fake_dict, df, 'tweet_text_stemming', 1)

stemming_fake_fig, stemming_fake_wordcloud = gera_nuvem_palavras(stemming_fake_dict)
#stemming_fake_wordcloud.to_file('wordcloud/corpus_labeled/iguais/wordcloud_tcc/03_exploratory_analysis_-_tweet_text_stemming_-_fake.png')
st.pyplot(stemming_fake_fig)

##### News
stemming_news_dict = dict()
stemming_news_dict = gera_frequencia_palavras(stemming_news_dict, df, 'tweet_text_stemming', -1)

stemming_news_fig, stemming_news_wordcloud = gera_nuvem_palavras(stemming_news_dict)
#stemming_news_wordcloud.to_file('wordcloud/corpus_labeled/iguais/wordcloud_tcc/03_exploratory_analysis_-_tweet_text_stemming_-_news.png')
st.pyplot(stemming_news_fig)

##### Opinion
stemming_opinion_dict = dict()
stemming_opinion_dict = gera_frequencia_palavras(stemming_opinion_dict, df, 'tweet_text_stemming', -1)

stemming_opinion_fig, stemming_opinion_wordcloud = gera_nuvem_palavras(stemming_opinion_dict)
#stemming_opinion_wordcloud.to_file('wordcloud/corpus_labeled/iguais/wordcloud_tcc/03_exploratory_analysis_-_tweet_text_stemming_-_opinion.png')
st.pyplot(stemming_opinion_fig)

#### Tweet_text_lemmatization
##### Fake News
lemmatization_fake_dict = dict()
lemmatization_fake_dict = gera_frequencia_palavras(lemmatization_fake_dict, df, 'tweet_text_lemmatization', 1)

lemmatization_fake_fig, lemmatization_fake_wordcloud = gera_nuvem_palavras(lemmatization_fake_dict)
#lemmatization_fake_wordcloud.to_file('wordcloud/corpus_labeled/iguais/wordcloud_tcc/03_exploratory_analysis_-_tweet_text_lemmatization_-_fake.png')
st.pyplot(lemmatization_fake_fig)

##### News
lemmatization_news_dict = dict()
lemmatization_news_dict = gera_frequencia_palavras(lemmatization_news_dict, df, 'tweet_text_lemmatization', -1)

lemmatization_news_fig, lemmatization_news_wordcloud = gera_nuvem_palavras(lemmatization_news_dict)
#lemmatization_news_wordcloud.to_file('wordcloud/corpus_labeled/iguais/wordcloud_tcc/03_exploratory_analysis_-_tweet_text_lemmatization_-_news.png')
st.pyplot(lemmatization_news_fig)

##### Opinion
lemmatization_opinion_dict = dict()
lemmatization_opinion_dict = gera_frequencia_palavras(lemmatization_opinion_dict, df, 'tweet_text_lemmatization', -1)

lemmatization_opinion_fig, lemmatization_opinion_wordcloud = gera_nuvem_palavras(lemmatization_opinion_dict)
#lemmatization_opinion_wordcloud.to_file('wordcloud/corpus_labeled/iguais/wordcloud_tcc/03_exploratory_analysis_-_tweet_text_lemmatization_-_opinion.png')
st.pyplot(lemmatization_opinion_fig)