import streamlit as st
import pandas as pd

st.set_page_config(
     page_title="Análises Estatísticas",
     page_icon="🌵",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={}
 )

st.experimental_singleton.clear()

st.header("Análises Estatísticas")

@st.experimental_singleton
def carrega_base(path):
    data = pd.read_csv(path, sep=",", low_memory=True)
    return data

df = carrega_base('models/08_statistics_fscore_base.csv')

st.subheader('Resultados estatísticos')
st.write('Na tabela a seguir são apresentos a média de desempenho, a partir da medida f-score e seu desvio padrão, alcançada por cada classificador em relação às classes news, opinion e fake.')
st.dataframe(df)
st.write('Ao analisar os resultados, é possível observar que dois modelos se destacaram sobre os demais. No que se refere a detecção de notícias verdadeiras (news) e notícias falsas (fake, o algoritmo Random Forest, fazendo uso da técnica de Oversampling, obteve os melhores resultados com, média 51,2\% de f-score e apresentando um desvio padrão de 2,8\% para a classe news. Já para a classe fake o modelo alcançou média de 33,2\% de f-score e desvio padrão de 5,8\%. Em relação à classe opinion, o modelo fazendo uso apenas dos algoritmo de Random Forest destacou-se dos demais ao alcançar um f-score com média de 84,3\% e desvio padrão igual a 1,8\%.')

st.subheader('Comparação entre modelos')
st.write('Com a finalidade de analisar a similaridade estatística dos modelos, os modelos com melhor desempenho foram comparados aos demais para cada classe. Portanto, o primeiro passo foi verificar a distribuição dos dados por meio do teste de normalidade de Shapiro-Wilk, em que a hipótese nula considera que os dados obedecem a uma distribuição normal. Após esta verificação, foi analisado se existe, de fato, diferença estatítica nos resultados de cada modelo, utilizando os testes T-Student ou de Wilcoxon 46 para as comparações. Os testes possuem 95\% de confiança, ou seja, consideraram um\α = 0.05.')

st.caption('News')
st.write('A imagem abaixo detalha as comparações realizadas entre o modelo Random Forest com oversampling em relação aos demais algoritmos, no que se refere à classe news. Durante os testes de normalidade não foi possível rejeitar a hipótese nula quanto a distribuição das f-scores de nenhum dos modelos, o teste de normalidade Shapiro-Wilk foi utilizado em todos os casos.')
st.image('models/boxplot/boxplot_news.png')
st.write('No gráfico acima é possível observar que o modelo Random Forest com Oversampling apesar de ter tido a maior média de f-score para a classe news possui desempenho superior a apenas 2 dos 7 modelos presentes nos experimentos')

st.caption('Opinion')
st.write('O gráfico seguinte detalha as comparações realizadas entre o modelo Random Forest em relação aos demais algoritmos, no que se refere à classe opinion. Assim como ocorreu nos experimentos anteriores, não foi possível rejeitar a hipótese nula durante os testes de normalidade, o teste de normalidade Shapiro-Wilk foi utilizado para avaliar cada modelo individualmente.')
st.image('models/boxplot/boxplot_opinion.png')
st.write('Como resultados da comparação do modelo de Random Forest em relação aos demais para a classe opinion, é possível observar que o modelo apresentou um desempenho melhor que 6 dos 7 modelos comparados. Essa classe obteve os melhores desempenhos gerais, pois em 6 dos 8 modelos analisados foi alcançada uma média de f-score igual ou superior a 80% (ver tabela 4). Este desempenho dos modelos na classe opinion, em parte, é devido à quantidade de exemplos rotulados com essa classe, que representa a maior parte das amostras de toda a base de dados.')

st.caption('Fake')
st.write('A seguir temos as comparações realizadas entre o modelo Random Forest com Oversampling em relação aos demais algoritmos para a classe fake. Durante os testes de normalidade. Para os modelos da classe fake foram rejeitadas a hipótese nula nos modelos de Ranfom Forest com Undersampling e de XGBoost, sendo assim, cada modelo teve seu desempenho avaliado por meio do teste de normalidade Shapiro-Wilk.')
st.image('models/boxplot/boxplot_fake.png')
st.write('O gráfico nos retorna que o modelo Random Forest com Oversampling, mesmo alcançando a melhor f-score média, apresentou um desempenho estatisticamente superior a apenas 1 dos 7 modelos comparados para a classe \textit{fake}.')

st.experimental_singleton.clear()

