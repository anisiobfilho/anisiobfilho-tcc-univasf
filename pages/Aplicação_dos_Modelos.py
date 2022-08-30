import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import nltk
import string
import gensim
from unidecode import unidecode
from sklearn.decomposition import PCA
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
#from cogroo4py.cogroo import Cogroo
import spacy
st.set_page_config(
     page_title="Aplicação dos Modelos",
     page_icon="🌵",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={}
 )

## FUNÇÕES DE PRÉ-PROCESSAMENTO
@st.experimental_memo
def carrega_base(path):
    data = pd.read_csv(path, low_memory=True)
    return data

df = carrega_base('data/corpus_labeled/iguais/bases_tcc/03_geracao_carcteristicas_base.csv')
df_internet = carrega_base('data/utils/abreviações_internet.csv')
df_estados = carrega_base("data/utils/abreviações_estados.csv")
df_vacinar = carrega_base("data/utils/flexões_vacinar.csv")
df_stopwords = carrega_base("data/utils/stopwords_internet_symboless.csv")

dict_internet = df_internet.set_index('sigla')['significado'].to_dict()
dict_estados = df_estados.set_index('sigla')['estado'].to_dict()
covidReplace = [ 'covid', 'covid-19', 'covid19', 'coronavirus', 'corona', 'virus' ]
vacinaReplace = [
                    'coronavac', 'astrazeneca', 'pfizer', 
                    'sputnik v', 'sputnik', 'sinovac', 
                    'oxford', 'moderna', 'butantan', 
                    'johnson', 'johnson&johnson', 'jnj', 
                    'fio cruz', 'fiocruz' 
                ]

nltk.download('stopwords')
stopWords = nltk.corpus.stopwords.words('portuguese')
spell = SpellChecker(language='pt')
#nltk.download('rslp')
#stemmer = nltk.stem.RSLPStemmer()
#cogroo = Cogroo()
#
nlp = spacy.load("pt_core_news_lg")

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={'hashtag', 'allcaps', 'elongated', 'repeated',
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter='twitter', 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector='twitter', 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def realiza_preproc(linha):
    ### Convertendo para string
    linha = str(linha)
    ### Removendo os '\n'
    linha = linha.replace('\n', ' ')
    ### Colocando as palavras em caixa baixa
    linha = linha.lower()
    ### Aplicando a correção gramatical do Spellchecker
    linha = spell.correction(linha)
    ### Aplicando o pre processamento do Ekprhasis
    linha = ' '.join(text_processor.pre_process_doc(linha))
    ### Removendo a pontuação da string
    linha = linha.translate(str.maketrans('', '', string.punctuation))
    ### Removendo a acentuação da string
    linha = unidecode(linha)
    ### Removendo espaços múltiplos
    linha = re.sub(r'\s+', ' ', linha)
    
    return linha

def tokenizador_nltk(linha):
    linha = word_tokenize(linha, language='portuguese')
    return linha

##Remoção de Stopwords utilizando NLTK:
def remove_stopwords_nltk(linha):
    nova_linha = []
    for parte in linha:
        if parte not in stopWords:
            nova_linha.append(parte)
        
    return nova_linha

def remove_stopwords_internet(linha):
    nova_linha = []
    for palavra in linha:
        if palavra not in df_stopwords.stopwords.to_list():
            nova_linha.append(palavra)
    
    return nova_linha

def troca_palavras(linha):
    nova_linha = []
    for palavra in linha:
        if palavra in dict_internet.keys():
            palavra = palavra.replace(palavra, dict_internet[palavra])
        if palavra in dict_estados.keys():
            palavra = palavra.replace(palavra, dict_estados[palavra])
        if palavra in df_vacinar.flexao.to_list():
            palavra = palavra.replace(palavra, 'vacina')
        if palavra in covidReplace:
            palavra = palavra.replace(palavra, 'covid')            
        if palavra in vacinaReplace:
            palavra = palavra.replace(palavra, 'vacina')
        nova_linha.append(palavra)

    return nova_linha

def lemmatization_spacy(linha):
    nova_linha = list()
    doc = nlp(str(linha))
    nova_linha = [token.lemma_ for token in doc if not token.is_punct]
        
    return nova_linha

def preprocessa_tweet(tweet):
    tweet = realiza_preproc(tweet)
    tweet = tokenizador_nltk(tweet)
    tweet = remove_stopwords_nltk(tweet)
    tweet = remove_stopwords_internet(tweet)
    tweet = troca_palavras(tweet)
    tweet = lemmatization_spacy(tweet)

    return tweet

def cria_modelo_word2vec(tweet): 
    tweet = preprocessa_tweet(tweet)
    modelo_word2vec = gensim.models.KeyedVectors.load('data/corpus_labeled/iguais/bases_tcc/05_word2vec_model_creation_base')

    class Sequencer():
    
        def __init__(self,
                    all_words,
                    max_words,
                    seq_len,
                    embedding_matrix
                    ):
            
            self.seq_len = seq_len
            self.embed_matrix = embedding_matrix
            """
            temp_vocab = Vocab which has all the unique words
            self.vocab = Our last vocab which has only most used N words.
        
            """
            temp_vocab = list(set(all_words))
            self.vocab = []
            self.word_cnts = {}
            """
            Now we'll create a hash map (dict) which includes words and their occurencies
            """
            for word in temp_vocab:
                # 0 does not have a meaning, you can add the word to the list
                # or something different.
                count = len([0 for w in all_words if w == word])
                self.word_cnts[word] = count
                counts = list(self.word_cnts.values())
                indexes = list(range(len(counts)))
            
            # Now we'll sort counts and while sorting them also will sort indexes.
            # We'll use those indexes to find most used N word.
            cnt = 0
            while cnt + 1 != len(counts):
                cnt = 0
                for i in range(len(counts)-1):
                    if counts[i] < counts[i+1]:
                        counts[i+1],counts[i] = counts[i],counts[i+1]
                        indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                    else:
                        cnt += 1
            
            for ind in indexes[:max_words]:
                self.vocab.append(temp_vocab[ind])
                        
        def textToVector(self,text):
            # First we need to split the text into its tokens and learn the length
            # If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
            # If it's longer than the max len we'll trim from the end.
            tokens = text#.split()
            len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
            vec = []
            for tok in tokens[:len_v]:
                try:
                    vec.append(self.embed_matrix[tok])
                except Exception as E:
                    pass
            
            last_pieces = self.seq_len - len(vec)
            for i in range(last_pieces):
                vec.append(np.zeros(100,))
            
            return np.asarray(vec).flatten()
    
    sequencer = Sequencer(all_words = [token for seq in df['tweet_text_stemming'] for token in seq],
            max_words = 1200,
            seq_len = 15,
            embedding_matrix = modelo_word2vec
            )
    x_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in df['tweet_text_stemming']])
    test_vec = sequencer.textToVector(tweet)
    
    pca_model = PCA(n_components=50)
    pca_model.fit(x_vecs)
    x_comps = pca_model.transform(test_vec.reshape(1,-1))   

    return x_comps
## MAIN
algoritmo = st.selectbox('Algoritmo',("Random Forest","XGBoost"))
oversampling = st.selectbox('Oversampling',(True, False)) 
undersampling = st.selectbox('Undersampling',(True, False)) 
tweet_text = st.text_input("Tweet")
resultado =""

if algoritmo == 'Random Forest':
    tag = 'RF'
elif algoritmo == 'XGBoost':
    tag = 'XGB'

modelo = jblid.load('models/model-'+tag+'_OV_'+str(oversampling)+'_UN_'+str(undersampling)+'.sav')

if st.button("Predict"): 
        resultado = modelo.predict(cria_modelo_word2vec(tweet_text))
        if resultado == 0:
            classe = 'Fake'
        elif resultado == 1:
            classe = 'Opinion'
        elif resultado == 2:
            classe = 'Fake'

        st.success('A classe deste tweet é: {}'.format(classe))
        print(resultado)