import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

st.set_page_config(
     page_title="Análise Exploratória",
     page_icon="🌵",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={}
 )

st.experimental_singleton.clear()

## Abertura de arquivo e criação do dataframe:
@st.experimental_singleton
def carrega_base(path):
    data = pd.read_csv(path, sep=",", low_memory=True)
    data.drop_duplicates(subset='tweet_id', keep='first', inplace=True)
    return data

df = carrega_base('data/corpus_labeled/iguais/bases_tcc/03_geracao_carcteristicas_base.csv')
#dfA1 = carrega_base("data/corpus_labeled/rotulaçao1[anisiofilho].csv")
#dfB1 = carrega_base("data/corpus_labeled/rotulaçao1[debora].csv")
dfA2 = carrega_base("data/corpus_labeled/rotulaçao2[anisiofilho].csv")
dfB2 = carrega_base("data/corpus_labeled/rotulaçao2[andre].csv")

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

def gera_frequencia_palavras1(dicionario_frequencia, dataframe, coluna):
    for index, row in dataframe.iterrows():
        row[coluna] = eval(row[coluna])
        for palavra in row[coluna]:
            if palavra not in dicionario_frequencia.keys():
                dicionario_frequencia[palavra] = 1
            else:
                dicionario_frequencia[palavra] += 1
    
    dicionario_frequencia = dict(sorted(dicionario_frequencia.items(), key=lambda item: item[1]))
    
    return dicionario_frequencia

### Contador de frequência de palavras:
def gera_frequencia_palavras2(dicionario_frequencia, dataframe, coluna, rotulo):
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

## MAIN
st.header("Análise Exploratória da Base de Dados")

st.write('O estudo de Filho et al. (2021) coletou uma base com um total de 10573 tweets, dois quais 1963 tweets foram rotulados e publicados, sendo: 320 tweets pertencentes à classe news; 275 pertencentes à classe fake e; 705 contidos na classe opinion. Os rotuladores obtiveram um índice Kappa = 0.47, que é considerado uma concordância moderada de avaliação. Neste trabalho, considerando o mesmo protocolo de anotação, foram acrescentados mais 2300 tweets ao corpus apresentado em Filho et al. (2021). Os rotuladores dos novos exemplos do corpus alcançaram um índice Kappa = 0.52, também considerado moderado.')
st.write('O gráfico abaixo apresenta a distribuição dos rótulos considerando a base de dados ampliada (Covid-19 Text Dataset - CTD).')

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

st.write('Foram incorporados a base de dados já existente apenas os rótulos que obtiveram concordância entre os dois anotadores dos novos exemplos. Sendo assim, o CTD foi contém 40 um total de 3600 exemplos, distribuídos entre as classes da seguinte forma: 487 para a classe fake; 714 contidos na classe new e; 2399 pertencentes à classe opinion.')


st.write('Com a finalidade de encontrar um padrão entre as publicações de notícias falsas, foram realizadas análises relacionadas a características como horário da postagem, localização e IDs dos usuários. Vale ressaltar que as próximas análises consideram sempre a base de dados ampliada (CTD).')

st.write('Os estudos relacionados aos horário de postagem consideraram as seguintes divisões por turno: Manhã compreende o período das 00h00min às 11h59min; Tarde, são as postagens realizadas das 12h00min às 17h59min e; Noite compreende o intervalo de horas das 18h00min às 23h59min. As postagens relacionadas à Covid-19 ocorrem, em geral, no período da noite, correspondendo a cerca de 86,11% do total de tweets, enquanto que o turno da manhã apresenta o menor período de engajamento, com aproximadamente 4,19\% do total de postagens da base. A distribuição das classes considerando as postagens por turno é apresentada no gráfico abaixo.')
df_turnos_rotulos = df[['time_shift', 'label_A']].value_counts().reset_index()
df_turnos_rotulos.rename(columns={'time_shift':'Turno','label_A':'Rótulo', 0:'Quantidade'}, inplace=True)
df_turnos_rotulos.sort_values(by='Quantidade', ascending=False, inplace=True)
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
st.write('O gráfico acima demonstra que todas as classes apresentam uma distribuição similar, em que a maioria dos tweets ocorreram no período da noite, seguidos pela tarde e por fim o turno da manhã. Para a classe fake, os tweets nortunos correspondem a 87,27\% dos 487 exemplos desta classe. Quando observamos a classe news vemos que dos seus 714 tweets, quase 80\% foram publicados no turno da noite. Para a classe opinion 87,83\% dos seus tweets foram postados a noite. Portanto, não foi possível identificar um horário em específico em que ocorre maior incidência de postagens de notícias falsas.')

st.write('Para a análise dos dados separando as postagens por Estados o processamento da localização dos usuários se deu da seguinte forma: primeiro converteu-se o texto em letras minúsculas, foram substituídas as siglas dos Estados para seu nome completo e também removeu-se os símbolos (acentos, pontuações e demais simbologias da língua). Por fim, utilizou-se a biblioteca Geopy para encontrar a localização mais precisa possível.')

st.write('A classificação se deu da seguinte forma:')
st.write('\t1. Os Estados que se encaixaram como sendo pertencentes ao Brasil, foram classificados com seus respectivos nomes;')
st.write('\t2. As localizações que só apresentavam o Brasil como referência, mas não apresentavam uma cidade ou Estado identificável foram identificados na categoria ‘regionnotdefined’;')
st.write('\t3. As localizações que não identificaram o Brasil como país de postagem, porém apresentavam algum outro país, foram categorizados como ‘notbrazilstate’;')
st.write('\t4. As localizações classificadas como ‘invalidlocation’ ou com conteúdo que não se encaixam nas categorias anteriores, como localizações com piadas ou conteúdo incompreensível foram classificadas como ‘invalidstate’.')

st.write('Após pré-processamento das localizações, foram encontrados os seguintes resultados: aproximadamente 47,47\% do total das postagens correspondem a Estados inválidos, enquanto que o Estado de São Paulo se sobressai com cerca de 13,19\% do total. Estados não pertencentes ao Brasil e Estados não definidos aparecem com cerca de 10\% e 7,44\% do total, respectivamente, na sequência temos o Rio de Janeiro, com postagens correspondentes a aproximadamente 6,08\%. Os demais Estados possuem menos de 5% dos exemplos rotulados nesta base de dados. O gráfico abaixo apresenta os resultados.')
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

st.write('A análise dos dados separando as postagens por região se deu de forma similizar ao processo utilizado para definir os Estados, a diferença consiste na definição de região, em que utilizou-se uma lista de Estados do Brasil para, a partir do Estado, identificarmos sua respectiva região. A classificação se deu da seguinte forma:')

st.write('\t1. As regiões identificadas como pertencentes ao Brasil, foram classificados com seus respectivos nomes;')
st.write('\t2. As localizações que só apresentavam o Brasil como localização, mas não apresentavam uma região detectável foram colocados na categoria ‘regionnotdefined’;')
st.write('\t3. As localizações que não identificaram o país como Brasil, porém como sendo algum outro país foram categorizados como ‘notbrazilregion’;')
st.write('\t4. As localizações classificadas como ‘invalidlocation’ ou com conteúdo que não se encaixa nas categorias anteriores, como localizações com piadas ou conteúdo incom- preensível, foram classificadas como ‘invalidregion’.')

st.write('Dessa forma, cerca de 47,47\% do total da base corresponde a regiões inválidas, enquanto que a região Sudeste se sobressai às demais com o aproximado de 22,92\% da base de dados, regiões não pertencentes ao Brasil aparecem em aproximadamente 10\% da base, enquanto que regiões indefinidas correspondem ao valor próximo de 7,44\% da base. As regiões Sul e Nordeste, aparecem com cerca de 5,64\% e 4,14\% dos tweets, respectivamente. As regiões Norte e Centro-Oeste contam com, sequencialmente, 1,5\% e aproximadamente 0,88\% do total de exemplos da base de dados.')
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
st.write('O gráfico acima detalha a distribuição das classes sob o ponto de vista das regiões. Neste ponto, um padrão é identificado, pois 22\% das publicações de pessoas que não adicionam suas localizações (invalidregion) pertencem à classe fake. Quanto as demais localizações, tem-se entre 9\% e 13\% das postagens contendo notícias falsas.')
st.write('Quanto as notícias verdadeiras, tem-se que 24,25\% dos tweets publicados com a localização ‘regionnotdefined’ são da classe news, seguido das publicações da região Nordeste, que são notícias verdadeiras em 23,49\% dos tuítes desta região. Ao analisar as postagens do Sul, tem-se notícias verdadeiras em 22,66\% das postagens. As demais regiões apresentam um percentual inferior a 20\% de tuítes com notícias verdadeiras. Por fim, é possível observar que a classe opinion é superior em todas as regiões, devido a natureza desbalanceada da base de dados coletada.')


st.write('A análise da base de dados separando as postagens pelos nomes de usuários foi feita recuperando-se o ‘user_screen_name’ a partir do id de cada tweet, esse processo foi feito utilizando-se um crawler/robô e a biblioteca Tweepy. Os dados recuperados foram armazenados na base de dados, o processamento foi feito analisando quais usuários possuem mais postagens dentro da base e depois disso foram escolhidos os dez usuários que mais postaram para as análises.')
df_usuarios_grafico = df.user_screen_name.value_counts().reset_index()[0:10]
df_usuarios_grafico.rename(columns={'index':'Nome de usuário','user_screen_name':'Quantidade'}, inplace=True)
df_usuarios_grafico.sort_values(by='Quantidade', ascending=True, inplace=True)
df_usuarios_grafico['Nome de usuário'] = df_usuarios_grafico['Nome de usuário'].map(ajusta_usuarios)
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
st.write('Ao considerarmos a separação dos tweets de cada usuário por classe, foi possível observar que alguns usuários postaram somente notícias verdadeiras, como exemplo tem-se o perfil do Instituto Butantan (butantanoficial). Também foi possível perceber que algumas contas postaram somente tweets considerados opiniões. Porém, ao analisarmos a classe fake, não foi possível observar nenhuma conta que publicou 100\% de notícias falsas. Algumas contas apresentam forte tendência para publicação de fake news, mas nem todos os tweets publicados eram notícias falsas. Esse é um fator importante de análise, pois alguns usuários podem variar entre notícias verdadeiras e falsas com a finalidade de ganhar a confiança dos demais.')

st.write('Com o intuito de analisar os termos com maior frequência na base de dados, foi gerada uma nuvem de palavras, conforme a figura 6, que destaca as palavras "vacina", "covid", "eficacia"e "brasil", como as mais frequentes no corpus.')
lemmatization_dict = dict()
lemmatization_dict = gera_frequencia_palavras1(lemmatization_dict, df, 'tweet_text_lemmatization')
lemmatization_fig, lemmatization_wordcloud = gera_nuvem_palavras(lemmatization_dict)
st.pyplot(lemmatization_fig)

st.write('Também foram analisadas as palavras mais frequentes de acordo com cada classe da base de dados. A figura abaixo realiza uma comparação entre as palavras mais relevantes tanto para a classe fake, figura a, quanto para a classe news, figura b e também para a classe opinion, figura c.')

coluna_lemmatization_fake, coluna_lemmatization_news, coluna_lemmatization_opinion = st.columns(3)
with coluna_lemmatization_fake:
    ##### Fake News
    st.caption('a')
    lemmatization_fake_dict = dict()
    lemmatization_fake_dict = gera_frequencia_palavras2(lemmatization_fake_dict, df, 'tweet_text_lemmatization', 1)

    lemmatization_fake_fig, lemmatization_fake_wordcloud = gera_nuvem_palavras(lemmatization_fake_dict)
    st.pyplot(lemmatization_fake_fig)

with coluna_lemmatization_news:
    ##### News
    st.caption('b')
    lemmatization_news_dict = dict()
    lemmatization_news_dict = gera_frequencia_palavras2(lemmatization_news_dict, df, 'tweet_text_lemmatization', -1)

    lemmatization_news_fig, lemmatization_news_wordcloud = gera_nuvem_palavras(lemmatization_news_dict)
    st.pyplot(lemmatization_news_fig)

with coluna_lemmatization_opinion:
    ##### Opinion
    st.caption('c')
    lemmatization_opinion_dict = dict()
    lemmatization_opinion_dict = gera_frequencia_palavras2(lemmatization_opinion_dict, df, 'tweet_text_lemmatization', -1)

    lemmatization_opinion_fig, lemmatization_opinion_wordcloud = gera_nuvem_palavras(lemmatization_opinion_dict)
    st.pyplot(lemmatization_opinion_fig)

st.write('Observando os resultados para a classe fake, é possível visualizar os termos "eficacia", "brasil", "vacina", "poder", "covid", "dose", "dado", "milhoes" e "contra", como os mais relevantes. Ao analisar os termos da classe news, as palavras com maior relevância são "vacina", "covid", "dose", "china", "eficacia", "fazer", "tomar", "doria", "querer"e "usar". Já para a classe opinon notamos palavras como "covid", "vacina", "eficacia", "milhoes", "brasil", "chinês", "dia", "brasil", "dose", "governo", "contra"e "anvisa". É possível perceber que alguns termos são apontados como muito frequentes tanto na classe de notícias verdadeiras quanto para as notícias falsas e também na classe de opinião. Nesse sentido, um ponto importante de análise está relacionado ao comportamento já observado quanto aos usuários que costumam publicar notícias falsas, pois estes também realizam publicações de notícias verdadeiras ou opiniões. Esses fatores são indicadores da dificuldade que os usuários de redes sociais podem encontrar para minerar o que seria uma notícia verdadeira ou falsa e especificar o que é apenas opinião.')

st.experimental_singleton.clear()