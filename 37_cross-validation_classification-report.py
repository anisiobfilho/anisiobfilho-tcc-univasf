##Essentials
import os
import csv
import numpy as np
from numpy.core.numeric import cross ##Numpy
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
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score, confusion_matrix
##Utils
import re
import unicodedata
from tqdm.auto import tqdm
import time
import timeit

start = timeit.default_timer()

df = pd.read_csv("data/data-twitter/training/rotulaçao[iguais]_complete.csv", sep=";")
X = pd.DataFrame()
X['tweet_text_stemmed']=df.loc[:,'tweet_text_stemmed'].apply(lambda x: np.str_(x))

y = pd.DataFrame()
y['label'] = df.loc[:,'label_A']
y.label += 1
y = np.ravel(y)

k = 10
kf = KFold(n_splits=k, random_state=42, shuffle=True)
#model = GaussianNB()

#X_train, X_test, y_train, y_test = train_test_split(X, y.label, test_size = 0.25, random_state = 10)

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

##Criando o modelo usando pipeline
model = imblearnPipeline(steps=[
    ('preprocessor', preprocessor),

    #DECISION TREE:
    ('decisiontree', DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=1, min_samples_split=3, random_state=None, splitter='random')),
    #DECISION TREE + OVER:
    #('oversampler', SMOTE()),
    #('decisiontree-ov', DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=4, random_state=1, splitter='best')),
    #DECISION TREE + OVER + UNDER:
    #('oversampler', SMOTE()),
    #('undersampler', RandomUnderSampler()),
    #('decisiontree-ov', DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=1, min_samples_split=5, random_state=None, splitter='random')),

    #MULTINOMIALNB:
    #('multinomialnb', MultinomialNB(alpha=0.5, class_prior=None)),
    #MULTINOMIALNB + OVER:
    #('oversampler', SMOTE()),
    #('multinomialnb+ov', MultinomialNB(alpha=0.5, class_prior=None)),
    #MULTINOMIALNB + OVER + UNDER:
    #('oversampler', SMOTE()),
    #('undersampler', RandomUnderSampler()),
    #('multinomialnb+ov+un', MultinomialNB(alpha=0.5, class_prior=None)),

    #SVC:
    #('svc', SVC(C=10, gamma=1, kernel='rbf', random_state=1)),
    #SVC + OVER:
    #('oversampler', SMOTE()),
    #('svc+ov', SVC(C=10, gamma=1, kernel='rbf', random_state=1)),
    #SVC + OVER + UNDER:
    #('oversampler', SMOTE()),
    #('undersampler', RandomUnderSampler()),
    #('svc+ov+un', SVC(C=10, gamma=1, kernel='rbf', random_state=1)),

    #RANDOM FOREST:
    #('randomforest', RandomForestClassifier(criterion='entropy', max_depth=None, max_features=9, min_samples_leaf=1, min_samples_split=10, random_state=1)),
    #RANDOM FOREST + OVER:
    #('oversampler', SMOTE()),
    #('randomforest+ov', RandomForestClassifier(criterion='entropy', max_depth=None, max_features=8, min_samples_leaf=1, min_samples_split=11, random_state=None)),
    #RANDOM FOREST + OVER + UNDER:
    #('oversampler', SMOTE()),
    #('undersampler', RandomUnderSampler()),
    #('randomforest+ov+un', RandomForestClassifier(criterion='gini', max_depth=None, max_features=9, min_samples_leaf=1, min_samples_split=9, random_state=1)),

    #ADABOOST:
    #('adaboost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=0.1, n_estimators=100)),
    #ADABOOST + OVER:
    #('oversampler', SMOTE()),
    #('adaboost+ov', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=1.0, n_estimators=500)),
    #ADABOOST + OVER + UNDER:
    #('oversampler', SMOTE()),
    #('undersampler', RandomUnderSampler()),
    #('adaboost+ov+un', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=1.0, n_estimators=500)),

    #XGBOOST:
    #('xgboost', XGBClassifier(colsample_bytree=0.8, eval_metric='mlogloss', gamma=1, learning_rate=0.01, max_depth=7, n_estimators=1000, nthread=4, num_class=3, objective='multi:softmax', reg_alpha=0.3, subsample=0.8, use_label_encoder=False)),
    #XGBOOST + OVER:
    #('oversampler', SMOTE()),
    #('xgboost+ov', XGBClassifier(colsample_bytree=0.8, eval_metric='mlogloss', gamma=0, learning_rate=0.01, max_depth=7, n_estimators=1000, nthread=4, num_class=3, objective='multi:softmax', reg_alpha=0.3, subsample=0.8, use_label_encoder=False)),
    #XGBOOST + OVER + UNDER:
    #('oversampler', SMOTE()),
    #('undersampler', RandomUnderSampler()),
    #('xgboost+ov+un', XGBClassifier(colsample_bytree=0.8, eval_metric='mlogloss', gamma=0, learning_rate=0.01, max_depth=7, n_estimators=1000, nthread=4, num_class=3, objective='multi:softmax', reg_alpha=0.3, subsample=0.8, use_label_encoder=False)),
])

split_num = 1
for train_index, test_index in kf.split(X):
    print('Split number:', split_num)   
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
    
    print(confusion_matrix(y_test, pred_values))
    print(classification_report(y_test, pred_values, target_names=['news', 'opinion', 'fake_news']))

    split_num = split_num + 1

end = timeit.default_timer()
print ('Duração: %f segundos' % (end - start))