# PIVIC - "Um modelo computacional para identificação de notícias falsas sobre a Covid-19 no Brasil"
# Code: Machine Learning - Supervised Learning
# Author: Anísio Pereira Batista Filho

##Essentials
import os
import csv
import numpy as np ##Numpy
import pandas as pd ##Pandas
##Sci-kit Learn
###Machine learning algorithms
from xgboost import XGBClassifier, XGBRegressor
from imblearn.pipeline import Pipeline as imblearnPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
##Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
###Pipeline, vectorizers and preprocessing
from sklearn.pipeline import Pipeline as sklearnPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
###Metrics
from sklearn.metrics import classification_report, accuracy_score
##Utils
import re
import unicodedata
from tqdm.auto import tqdm
import time
import timeit

import xgboost

start = timeit.default_timer()

#Colunas: tweet_text;tweet_text_lower;tweet_text_stemmed;tweet_text_lemmatized;tweet_text_spellchecked;tweet_text_spellchecked_lower;tweet_text_spellchecked_stemmed;tweet_text_spellchecked_lemmatized
##lendo o dataset
df = pd.read_csv("data/data-twitter/training/rotulaçao[iguais]_complete.csv", sep=";")
#X=df[['time_shift', 'region_location']] #'state_location',
X = pd.DataFrame()
#X['tweet_text_stemmed']=df['tweet_text_stemmed'].apply(lambda x: np.str_(x))
X['tweet_text_stemmed']=df.loc[:,'tweet_text_stemmed'].apply(lambda x: np.str_(x))
#X = pd.DataFrame()
#X=df[['tweet_text_stemmed']].astype(str)
#X=df[['time_shift', 'region_location']] #'state_location',
#text_list = df.loc[:,'tweet_text_stemmed'].apply(lambda x: np.str_(x))
#X['tweet_text_stemmed'] = text_list

#X['created_at'] = pd.to_numeric(df.created_at.str.replace('-','').replace(':','').replace('+','').replace(' ',''))
y = pd.DataFrame()
y['label'] = df.loc[:,'label_A']
y.label += 1

##Separando dados de treinamento e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y.label, test_size = 0.25, random_state = 10)

##Pipeline para o MinMaxScaler()
minmax_transformer = sklearnPipeline(steps=[
    ('imputer', MinMaxScaler(feature_range=(0, 1)))
])

##Pipeline para simpleimputer()
num_transformer = sklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

##Pipeline para OneHotEncoder()
cat_transformer = sklearnPipeline(steps=[
    ('one-hot encoder', OneHotEncoder())
])

##Pipeline para TfidfVectorizer()
tfidf_transfomer = sklearnPipeline(steps=[
    ('tf-idf', TfidfVectorizer())
])

##Compondo os pré-processadores
preprocessor = ColumnTransformer(transformers=[
    #('minmax', minmax_transformer, ['created_at']),
    #('num', num_transformer, ['amount']),
    ('tf-idf', tfidf_transfomer, 'tweet_text_stemmed'),
    #('cat', cat_transformer, ['time_shift', 'region_location'])
    ],
    #remainder='passthrough'
    )


vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(X_train['tweet_text_stemmed']).toarray()
caracteristicas = vectorizer.get_feature_names()

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)

print("XGBoost + Oversampling")
parameters = {  'nthread': [4], #when use hyperthread, xgboost may become slower
                'learning_rate': [0.01], #so called `eta` value
                'max_depth': [7],
                #'min_child_weight': [11],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'n_estimators': [1000], #number of trees, change it to 1000 for better results
                #'missing': [-999],
                #'seed': [1337],
                #'booster': ['gbdt'],
                #'metric': ['multiclass'],
                'eval_metric': ['mlogloss'],
                #'silent': [False], 
                #'scale_pos_weight': [1],  
                #'subsample': [0.8],
                'objective': ['multi:softmax'], 
                'reg_alpha': [0.3],
                'gamma': [0],#], 1],
                'use_label_encoder': [False],
                'num_class': [3]
            }
model = XGBClassifier(parameters)
#print("Random Forest + Oversampling")
#model = RandomForestClassifier(criterion='entropy', max_depth=None, max_features=8, min_samples_leaf=1, min_samples_split=11, random_state=None)
model.fit(x_train, y_train)

mdg_caracteristicas = model.feature_importances_

lista = [x for x in mdg_caracteristicas if x > 0]

print('Tamanho da lista de caracteristicas com MDG > 0 =', len(lista))

feature_importances = pd.DataFrame(mdg_caracteristicas,
                                   index = caracteristicas,
                                   columns=['importance']).sort_values('importance',ascending=False)


index_feature_importances = pd.DataFrame(mdg_caracteristicas,
                                   index = range(len(caracteristicas)),
                                   columns=['importance']).sort_values('importance',ascending=False)


labels_features = feature_importances['importance'].index[:20]
indices_features = index_feature_importances['importance'].index[:20]
mdg_features = feature_importances['importance'].values[:20]
description = ["description" for i in range(len(mdg_features))]

data = {"Variable": labels_features, "Description": indices_features, "MDG": mdg_features}
pd.DataFrame(data).to_csv("top20-xgb_ov.csv")
#pd.DataFrame(data).to_csv("top20-rf_ov.csv")

end = timeit.default_timer()
print ('Duração: %f segundos' % (end - start))