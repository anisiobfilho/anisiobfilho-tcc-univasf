import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import shapiro, normaltest, anderson, levene, ttest_ind, wilcoxon

import time
import timeit

start = timeit.default_timer()

#FUNCTIONS:
##STATISTICS TESTS:
def shapiro_wilk_test(data):
    # normality test
    stat, p = shapiro(data)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tSample looks Gaussian (fail to reject H0)')
    else:
        print('\tSample does not look Gaussian (reject H0)')
    return p

def dagostino_test(data):
    # normality test
    stat, p = normaltest(data)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tSample looks Gaussian (fail to reject H0)')
    else:
        print('\tSample does not look Gaussian (reject H0)')

def anderson_darling_test(data):
    # normality test
    result = anderson(data)
    print('\tStatistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('\t%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('\t%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

def levene_test(data1, data2):
    # normality test
    stat, p = levene(data1, data2)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tThe difference between the means is not statistically significant (do not reject H0)')
    else:
        print('\tThe difference between the means is statistically significant (reject H0)')
    return

def tstudent_test(data1, data2):
    print("\tT-Student Test:")
    # normality test
    stat, p = ttest_ind(data1, data2, equal_var=True)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tThe difference between the means is not statistically significant (do not reject H0)')
    else:
        print('\tThe difference between the means is statistically significant (reject H0)')
    return

def wilcoxon_test(data1, data2):
    print("\tWilcoxon Test:")
    # normality test
    stat, p = wilcoxon(data1, data2)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tThe difference between the means is not statistically significant (do not reject H0)')
    else:
        print('\tThe difference between the means is statistically significant (reject H0)')
    return


def perform_fulltest(data1, data2, alpha1, alpha2):
    if alpha1 > 0.05 and alpha2 > 0.05:
        tstudent_test(data1, data2)
    else:
        wilcoxon_test(data1, data2)

    return


##MAIN:
fscore_news    = [0.42, 0.51, 0.45, 0.50, 0.52, 0.50, 0.52, 0.51, 0.53, 0.58, 0.57, 0.60, 0.53, 0.57, 0.53, 0.51, 0.50, 0.51]
fscore_opinion = [0.79, 0.79, 0.80, 0.81, 0.72, 0.70, 0.84, 0.84, 0.84, 0.85, 0.86, 0.87, 0.79, 0.85, 0.83, 0.85, 0.86, 0.86]
fscore_fake    = [0.47, 0.41, 0.45, 0.35, 0.41, 0.38, 0.44, 0.42, 0.44, 0.33, 0.43, 0.40, 0.45, 0.41, 0.48, 0.50, 0.43, 0.49]
weighted_avg   = [0.65, 0.67, 0.67, 0.72, 0.58, 0.55, 0.75, 0.75, 0.76, 0.79, 0.78, 0.79, 0.68, 0.74, 0.73, 0.76, 0.75, 0.75]

dt_news    = [0.50, 0.51, 0.62, 0.43, 0.47, 0.56, 0.52, 0.53, 0.43, 0.56]
dt_opinion = [0.71, 0.80, 0.80, 0.76, 0.81, 0.81, 0.84, 0.70, 0.72, 0.81]
dt_fake    = [0.43, 0.43, 0.30, 0.36, 0.45, 0.33, 0.34, 0.30, 0.36, 0.47]

dt_ov_news    = [0.52, 0.48, 0.55, 0.35, 0.47, 0.45, 0.42, 0.55, 0.45, 0.42]
dt_ov_opinion = [0.66, 0.73, 0.74, 0.75, 0.81, 0.74, 0.81, 0.74, 0.71, 0.77]
dt_ov_fake    = [0.50, 0.36, 0.38, 0.42, 0.47, 0.38, 0.56, 0.17, 0.40, 0.41]

dt_ov_un_news    = [0.53, 0.43, 0.63, 0.46, 0.55, 0.54, 0.47, 0.54, 0.51, 0.52]
dt_ov_un_opinion = [0.75, 0.77, 0.80, 0.75, 0.82, 0.77, 0.83, 0.68, 0.73, 0.74]
dt_ov_un_fake    = [0.32, 0.41, 0.36, 0.40, 0.45, 0.30, 0.34, 0.44, 0.43, 0.42]

mnb_news    = [0.52, 0.50, 0.69, 0.36, 0.56, 0.63, 0.51, 0.68, 0.59, 0.54]
mnb_opinion = [0.75, 0.75, 0.80, 0.79, 0.84, 0.81, 0.83, 0.80, 0.78, 0.79]
mnb_fake    = [0.36, 0.38, 0.13, 0.34, 0.47, 0.21, 0.12, 0.31, 0.43, 0.52]

mnb_ov_news    = [0.53, 0.57, 0.65, 0.42, 0.55, 0.48, 0.52, 0.65, 0.62, 0.59]
mnb_ov_opinion = [0.56, 0.67, 0.69, 0.63, 0.73, 0.71, 0.73, 0.69, 0.64, 0.75]
mnb_ov_fake    = [0.38, 0.49, 0.33, 0.33, 0.46, 0.48, 0.43, 0.35, 0.49, 0.56]

mnb_ov_un_news    = [0.51, 0.52, 0.63, 0.47, 0.52, 0.53, 0.47, 0.65, 0.64, 0.56]
mnb_ov_un_opinion = [0.58, 0.64, 0.69, 0.62, 0.68, 0.73, 0.74, 0.67, 0.67, 0.78]
mnb_ov_un_fake    = [0.44, 0.50, 0.34, 0.36, 0.51, 0.44, 0.48, 0.34, 0.44, 0.52]

svc_news    = [0.62, 0.56, 0.72, 0.57, 0.58, 0.59, 0.48, 0.64, 0.60, 0.62]
svc_opinion = [0.79, 0.78, 0.83, 0.82, 0.87, 0.82, 0.84, 0.79, 0.78, 0.77]
svc_fake    = [0.49, 0.40, 0.39, 0.43, 0.67, 0.19, 0.49, 0.34, 0.46, 0.52]

svc_ov_news    = [0.64, 0.59, 0.72, 0.56, 0.63, 0.60, 0.47, 0.62, 0.61, 0.62]
svc_ov_opinion = [0.79, 0.78, 0.83, 0.82, 0.87, 0.82, 0.83, 0.79, 0.78, 0.77]
svc_ov_fake    = [0.50, 0.41, 0.39, 0.44, 0.64, 0.24, 0.42, 0.33, 0.50, 0.52]

svc_ov_un_news    = [0.64, 0.60, 0.70, 0.57, 0.60, 0.58, 0.47, 0.62, 0.60, 0.62]
svc_ov_un_opinion = [0.79, 0.78, 0.82, 0.82, 0.87, 0.82, 0.84, 0.79, 0.78, 0.77]
svc_ov_un_fake    = [0.50, 0.44, 0.39, 0.43, 0.62, 0.24, 0.45, 0.29, 0.46, 0.52]

rf_news    = [0.57, 0.56, 0.62, 0.46, 0.58, 0.58, 0.44, 0.65, 0.52, 0.46]
rf_opinion = [0.78, 0.77, 0.81, 0.79, 0.85, 0.80, 0.80, 0.81, 0.78, 0.76]
rf_fake    = [0.33, 0.47, 0.26, 0.38, 0.44, 0.21, 0.12, 0.26, 0.26, 0.45]

rf_ov_news    = [0.57, 0.62, 0.68, 0.59, 0.70, 0.66, 0.62, 0.64, 0.58, 0.55]
rf_ov_opinion = [0.77, 0.79, 0.83, 0.84, 0.89, 0.83, 0.85, 0.79, 0.81, 0.79]
rf_ov_fake    = [0.36, 0.47, 0.41, 0.39, 0.52, 0.33, 0.33, 0.41, 0.53, 0.59]

rf_ov_un_news    = [0.52, 0.57, 0.69, 0.50, 0.68, 0.62, 0.55, 0.64, 0.54, 0.58]
rf_ov_un_opinion = [0.79, 0.79, 0.83, 0.82, 0.87, 0.82, 0.84, 0.80, 0.79, 0.81]
rf_ov_un_fake    = [0.49, 0.47, 0.29, 0.40, 0.48, 0.39, 0.28, 0.29, 0.47, 0.52]

ada_news    = [0.52, 0.53, 0.60, 0.37, 0.58, 0.49, 0.40, 0.59, 0.49, 0.58]
ada_opinion = [0.79, 0.79, 0.82, 0.74, 0.84, 0.83, 0.80, 0.74, 0.73, 0.77]
ada_fake    = [0.41, 0.48, 0.37, 0.44, 0.47, 0.47, 0.51, 0.42, 0.40, 0.61]

ada_ov_news    = [0.62, 0.52, 0.61, 0.49, 0.63, 0.59, 0.50, 0.57, 0.41, 0.41]
ada_ov_opinion = [0.82, 0.77, 0.82, 0.80, 0.85, 0.86, 0.77, 0.72, 0.78, 0.74]
ada_ov_fake    = [0.42, 0.50, 0.34, 0.47, 0.49, 0.44, 0.31, 0.46, 0.50, 0.46]

ada_ov_un_news    = [0.61, 0.52, 0.58, 0.51, 0.62, 0.60, 0.60, 0.56, 0.44, 0.63]
ada_ov_un_opinion = [0.72, 0.78, 0.83, 0.81, 0.86, 0.84, 0.82, 0.74, 0.76, 0.82]
ada_ov_un_fake    = [0.52, 0.50, 0.40, 0.43, 0.57, 0.37, 0.36, 0.36, 0.57, 0.61]

xgb_news    = [0.58, 0.48, 0.64, 0.42, 0.52, 0.62, 0.52, 0.64, 0.52, 0.63]
xgb_opinion = [0.81, 0.81, 0.82, 0.79, 0.86, 0.86, 0.83, 0.79, 0.77, 0.84]
xgb_fake    = [0.46, 0.46, 0.39, 0.55, 0.44, 0.52, 0.40, 0.40, 0.45, 0.57]

xgb_ov_news    = [0.52, 0.47, 0.60, 0.44, 0.58, 0.52, 0.54, 0.56, 0.61, 0.67]
xgb_ov_opinion = [0.77, 0.82, 0.81, 0.81, 0.85, 0.82, 0.84, 0.77, 0.79, 0.85]
xgb_ov_fake    = [0.42, 0.53, 0.42, 0.50, 0.45, 0.44, 0.49, 0.34, 0.51, 0.65]

xgb_ov_un_news    = [0.54, 0.52, 0.66, 0.43, 0.49, 0.53, 0.54, 0.53, 0.56, 0.62]
xgb_ov_un_opinion = [0.79, 0.81, 0.84, 0.78, 0.85, 0.83, 0.83, 0.77, 0.79, 0.83]
xgb_ov_un_fake    = [0.41, 0.54, 0.39, 0.49, 0.44, 0.45, 0.35, 0.28, 0.41, 0.67]

print("\n")
print("Mean:")
print("\n")
print("Decision Tree:")
print("News: %.2f" % (np.mean(dt_news)))
print("Opinion: %.2f" % (np.mean(dt_opinion)))
print("Fake: %.2f" % (np.mean(dt_fake)))
print("\n")
print("DT + OV:")
print("News: %.2f" % (np.mean(dt_ov_news)))
print("Opinion %.2f" % (np.mean(dt_ov_opinion)))
print("Fake %.2f" % (np.mean(dt_ov_fake)))
print("\n")
print("DT + OV + UN:")
print("News %.2f" % (np.mean(dt_ov_un_news)))
print("Opinion %.2f" % (np.mean(dt_ov_un_opinion)))
print("Fake %.2f" % (np.mean(dt_ov_un_fake)))
print("\n")
print("Multinomial Naive Bayes:")
print("News %.2f" % (np.mean(mnb_news)))
print("Opinion %.2f" % (np.mean(mnb_opinion)))
print("Fake %.2f" % (np.mean(mnb_fake)))
print("\n")
print("MNB + OV:")
print("News %.2f" % (np.mean(mnb_ov_news)))
print("Opinion %.2f" % (np.mean(mnb_ov_opinion)))
print("Fake %.2f" % (np.mean(mnb_ov_fake)))
print("\n")
print("MNB + OV + UN:")
print("News %.2f" % (np.mean(mnb_ov_un_news)))
print("Opinion %.2f" % (np.mean(mnb_ov_un_opinion)))
print("Fake %.2f" % (np.mean(mnb_ov_un_fake)))
print("\n")
print("SVC:")
print("News %.2f" % (np.mean(svc_news)))
print("Opinion %.2f" % (np.mean(svc_opinion)))
print("Fake %.2f" % (np.mean(svc_fake)))
print("\n")
print("SVC + OV:")
print("News %.2f" % (np.mean(svc_ov_news)))
print("Opinion %.2f" % (np.mean(svc_ov_opinion)))
print("Fake %.2f" % (np.mean(svc_ov_fake)))
print("\n")
print("SVC + OV + UN:")
print("News %.2f" % (np.mean(svc_ov_un_news)))
print("Opinion %.2f" % (np.mean(svc_ov_un_opinion)))
print("Fake %.2f" % (np.mean(svc_ov_un_fake)))
print("\n")
print("Random Forest:")
print("News %.2f" % (np.mean(rf_news)))
print("Opinion %.2f" % (np.mean(rf_opinion)))
print("Fake %.2f" % (np.mean(rf_fake)))
print("\n")
print("RF + OV:")
print("News %.2f" % (np.mean(rf_ov_news)))
print("Opinion %.2f" % (np.mean(rf_ov_opinion)))
print("Fake %.2f" % (np.mean(rf_ov_fake)))
print("\n")
print("RF + OV + UN:")
print("News %.2f" % (np.mean(rf_ov_un_news)))
print("Opinion %.2f" % (np.mean(rf_ov_un_opinion)))
print("Fake %.2f" % (np.mean(rf_ov_un_fake)))
print("\n")
print("Adaboost:")
print("News %.2f" % (np.mean(ada_news)))
print("Opinion %.2f" % (np.mean(ada_opinion)))
print("Fake %.2f" % (np.mean(ada_fake)))
print("\n")
print("Adaboost + OV:")
print("News %.2f" % (np.mean(ada_ov_news)))
print("Opinion %.2f" % (np.mean(ada_ov_opinion)))
print("Fake %.2f" % (np.mean(ada_ov_fake)))
print("\n")
print("Adaboost + OV + UN:")
print("News %.2f" % (np.mean(ada_ov_un_news)))
print("Opinion %.2f" % (np.mean(ada_ov_un_opinion)))
print("Fake %.2f" % (np.mean(ada_ov_un_fake)))
print("\n")
print("XGBoost:")
print("News %.2f" % (np.mean(xgb_news)))
print("Opinion %.2f" % (np.mean(xgb_opinion)))
print("Fake %.2f" % (np.mean(xgb_fake)))
print("\n")
print("XGBoost + OV:")
print("News %.2f" % (np.mean(xgb_ov_news)))
print("Opinion %.2f" % (np.mean(xgb_ov_opinion)))
print("Fake %.2f" % (np.mean(xgb_ov_fake)))
print("\n")
print("XGBoost + OV + UN:")
print("News %.2f" % (np.mean(xgb_ov_un_news)))
print("Opinion %.2f" % (np.mean(xgb_ov_un_opinion)))
print("Fake %.2f" % (np.mean(xgb_ov_un_fake)))
print("\n")
print("News %.2f" % (np.mean(fscore_news)))
print("Opinion %.2f" % (np.mean(fscore_opinion)))
print("Fake News %.2f" % (np.mean(fscore_fake)))
print("Weighted Average %.2f" % (np.mean(weighted_avg)))

print("\n")
print("Standard Deviaton:")
print("\n")
print("Decision Tree:")
print("News %.2f" % (np.std(dt_news)))
print("Opinion %.2f" % (np.std(dt_opinion)))
print("Fake %.2f" % (np.std(dt_fake)))
print("\n")
print("DT + OV:")
print("News %.2f" % (np.std(dt_ov_news)))
print("Opinion %.2f" % (np.std(dt_ov_opinion)))
print("Fake %.2f" % (np.std(dt_ov_fake)))
print("\n")
print("DT + OV + UN:")
print("News %.2f" % (np.std(dt_ov_un_news)))
print("Opinion %.2f" % (np.std(dt_ov_un_opinion)))
print("Fake %.2f" % (np.std(dt_ov_un_fake)))
print("\n")
print("Multinomial Naive Bayes:")
print("News %.2f" % (np.std(mnb_news)))
print("Opinion %.2f" % (np.std(mnb_opinion)))
print("Fake %.2f" % (np.std(mnb_fake)))
print("\n")
print("MNB + OV:")
print("News %.2f" % (np.std(mnb_ov_news)))
print("Opinion %.2f" % (np.std(mnb_ov_opinion)))
print("Fake %.2f" % (np.std(mnb_ov_fake)))
print("\n")
print("MNB + OV + UN:")
print("News %.2f" % (np.std(mnb_ov_un_news)))
print("Opinion %.2f" % (np.std(mnb_ov_un_opinion)))
print("Fake %.2f" % (np.std(mnb_ov_un_fake)))
print("\n")
print("SVC:")
print("News %.2f" % (np.std(svc_news)))
print("Opinion %.2f" % (np.std(svc_opinion)))
print("Fake %.2f" % (np.std(svc_fake)))
print("\n")
print("SVC + OV:")
print("News %.2f" % (np.std(svc_ov_news)))
print("Opinion %.2f" % (np.std(svc_ov_opinion)))
print("Fake %.2f" % (np.std(svc_ov_fake)))
print("\n")
print("SVC + OV + UN:")
print("News %.2f" % (np.std(svc_ov_un_news)))
print("Opinion %.2f" % (np.std(svc_ov_un_opinion)))
print("Fake %.2f" % (np.std(svc_ov_un_fake)))
print("\n")
print("Random Forest:")
print("News %.2f" % (np.std(rf_news)))
print("Opinion %.2f" % (np.std(rf_opinion)))
print("Fake %.2f" % (np.std(rf_fake)))
print("\n")
print("RF + OV:")
print("News %.2f" % (np.std(rf_ov_news)))
print("Opinion %.2f" % (np.std(rf_ov_opinion)))
print("Fake %.2f" % (np.std(rf_ov_fake)))
print("\n")
print("RF + OV + UN:")
print("News %.2f" % (np.std(rf_ov_un_news)))
print("Opinion %.2f" % (np.std(rf_ov_un_opinion)))
print("Fake %.2f" % (np.std(rf_ov_un_fake)))
print("\n")
print("Adaboost:")
print("News %.2f" % (np.std(ada_news)))
print("Opinion %.2f" % (np.std(ada_opinion)))
print("Fake %.2f" % (np.std(ada_fake)))
print("\n")
print("Adaboost + OV:")
print("News %.2f" % (np.std(ada_ov_news)))
print("Opinion %.2f" % (np.std(ada_ov_opinion)))
print("Fake %.2f" % (np.std(ada_ov_fake)))
print("\n")
print("Adaboost + OV + UN:")
print("News %.2f" % (np.std(ada_ov_un_news)))
print("Opinion %.2f" % (np.std(ada_ov_un_opinion)))
print("Fake %.2f" % (np.std(ada_ov_un_fake)))
print("\n")
print("XGBoost:")
print("News %.2f" % (np.std(xgb_news)))
print("Opinion %.2f" % (np.std(xgb_opinion)))
print("Fake %.2f" % (np.std(xgb_fake)))
print("\n")
print("XGBoost + OV:")
print("News %.2f" % (np.std(xgb_ov_news)))
print("Opinion %.2f" % (np.std(xgb_ov_opinion)))
print("Fake %.2f" % (np.std(xgb_ov_fake)))
print("\n")
print("XGBoost + OV + UN:")
print("News %.2f" % (np.std(xgb_ov_un_news)))
print("Opinion %.2f" % (np.std(xgb_ov_un_opinion)))
print("Fake %.2f" % (np.std(xgb_ov_un_fake)))
print("\n")
print("News %.2f" % (np.std(fscore_news)))
print("Opinion %.2f" % (np.std(fscore_opinion)))
print("Fake News %.2f" % (np.std(fscore_fake)))
print("Weighted Average %.2f" % (np.std(weighted_avg)))

#Normality Tests:
print("\n")
print("Normality Tests:")

print("Shapiro-Wilk Tests:")
print("\n")
print("Decision Tree:")
print("News:")
dt_news_alpha = shapiro_wilk_test(dt_news)
print("Opinion:")
dt_opinion_alpha = shapiro_wilk_test(dt_opinion)
print("Fake:")
dt_fake_alpha = shapiro_wilk_test(dt_fake)

print("\n")
print("DT + Oversampling:")
print("News:")
dt_ov_news_alpha = shapiro_wilk_test(dt_ov_news)
print("Opinion:")
dt_ov_opinion_alpha = shapiro_wilk_test(dt_ov_opinion)
print("Fake:")
dt_ov_fake_alpha = shapiro_wilk_test(dt_ov_fake)

print("\n")
print("DT + Oversampling + Undersampling:")
print("News:")
dt_ov_un_news_alpha = shapiro_wilk_test(dt_ov_un_news)
print("Opinion:")
dt_ov_un_opinion_alpha = shapiro_wilk_test(dt_ov_un_opinion)
print("Fake:")
dt_ov_un_fake_alpha = shapiro_wilk_test(dt_ov_un_fake)

print("\n")
print("Multinomial Naive Bayes:")
print("News:")
mnb_news_alpha = shapiro_wilk_test(mnb_news)
print("Opinion:")
mnb_opinion_alpha = shapiro_wilk_test(mnb_opinion)
print("Fake:")
mnb_fake_alpha = shapiro_wilk_test(mnb_fake)

print("\n")
print("MNB + Oversampling:")
print("News:")
mnb_ov_news_alpha = shapiro_wilk_test(mnb_ov_news)
print("Opinion:")
mnb_ov_opinion_alpha = shapiro_wilk_test(mnb_ov_opinion)
print("Fake:")
mnb_ov_fake_alpha = shapiro_wilk_test(mnb_ov_fake)

print("\n")
print("MNB + Oversampling + Undersampling:")
print("News:")
mnb_ov_un_news_alpha = shapiro_wilk_test(mnb_ov_un_news)
print("Opinion:")
mnb_ov_un_opinion_alpha = shapiro_wilk_test(mnb_ov_un_opinion)
print("Fake:")
mnb_ov_un_fake_alpha = shapiro_wilk_test(mnb_ov_un_fake)

print("\n")
print("SVC:")
print("News:")
svc_news_alpha = shapiro_wilk_test(svc_news)
print("Opinion:")
svc_opinion_alpha = shapiro_wilk_test(svc_opinion)
print("Fake:")
svc_fake_alpha = shapiro_wilk_test(svc_fake)

print("\n")
print("SVC + Oversampling:")
print("News:")
svc_ov_news_alpha = shapiro_wilk_test(svc_ov_news)
print("Opinion:")
svc_ov_opinion_alpha = shapiro_wilk_test(svc_ov_opinion)
print("Fake:")
svc_ov_fake_alpha = shapiro_wilk_test(svc_ov_fake)

print("\n")
print("SVC + Oversampling + Undersampling:")
print("News:")
svc_ov_un_news_alpha = shapiro_wilk_test(svc_ov_un_news)
print("Opinion:")
svc_ov_un_opinion_alpha = shapiro_wilk_test(svc_ov_un_opinion)
print("Fake:")
svc_ov_un_fake_alpha = shapiro_wilk_test(svc_ov_un_fake)

print("\n")
print("Random Forest:")
print("News:")
rf_news_alpha = shapiro_wilk_test(rf_news)
print("Opinion:")
rf_opinion_alpha = shapiro_wilk_test(rf_opinion)
print("Fake:")
rf_fake_alpha = shapiro_wilk_test(rf_fake)

print("\n")
print("RF + Oversampling:")
print("News:")
rf_ov_news_alpha = shapiro_wilk_test(rf_ov_news)
print("Opinion:")
rf_ov_opinion_alpha = shapiro_wilk_test(rf_ov_opinion)
print("Fake:")
rf_ov_fake_alpha = shapiro_wilk_test(rf_ov_fake)

print("\n")
print("RF + Oversampling + Undersampling:")
print("News:")
rf_ov_un_news_alpha = shapiro_wilk_test(rf_ov_un_news)
print("Opinion:")
rf_ov_un_opinion_alpha = shapiro_wilk_test(rf_ov_un_opinion)
print("Fake:")
rf_ov_un_fake_alpha = shapiro_wilk_test(rf_ov_un_fake)

print("\n")
print("Adaboost:")
print("News:")
ada_news_alpha = shapiro_wilk_test(ada_news)
print("Opinion:")
ada_opinion_alpha = shapiro_wilk_test(ada_opinion)
print("Fake:")
ada_fake_alpha = shapiro_wilk_test(ada_fake)

print("\n")
print("Adaboost + Oversampling:")
print("News:")
ada_ov_news_alpha = shapiro_wilk_test(ada_ov_news)
print("Opinion:")
ada_ov_opinion_alpha = shapiro_wilk_test(ada_ov_opinion)
print("Fake:")
ada_ov_fake_alpha = shapiro_wilk_test(ada_ov_fake)

print("\n")
print("Adaboost + Oversampling + Undersampling:")
print("News:")
ada_ov_un_news_alpha = shapiro_wilk_test(ada_ov_un_news)
print("Opinion:")
ada_ov_un_opinion_alpha = shapiro_wilk_test(ada_ov_un_opinion)
print("Fake:")
ada_ov_un_fake_alpha = shapiro_wilk_test(ada_ov_un_fake)

print("\n")
print("XGBoost:")
print("News:")
xgb_news_alpha = shapiro_wilk_test(xgb_news)
print("Opinion:")
xgb_opinion_alpha = shapiro_wilk_test(xgb_opinion)
print("Fake:")
xgb_fake_alpha = shapiro_wilk_test(xgb_fake)

print("\n")
print("XGBoost + Oversampling:")
print("News:")
xgb_ov_news_alpha = shapiro_wilk_test(xgb_ov_news)
print("Opinion:")
xgb_ov_opinion_alpha = shapiro_wilk_test(xgb_ov_opinion)
print("Fake:")
xgb_ov_fake_alpha = shapiro_wilk_test(xgb_ov_fake)

print("\n")
print("XGBoost + Oversampling + Undersampling:")
print("News:")
xgb_ov_un_news_alpha = shapiro_wilk_test(xgb_ov_un_news)
print("Opinion:")
xgb_ov_un_opinion_alpha = shapiro_wilk_test(xgb_ov_un_opinion)
print("Fake:")
xgb_ov_un_fake_alpha = shapiro_wilk_test(xgb_ov_un_fake)

print("\n")
print("News:")
news_alpha = shapiro_wilk_test(fscore_news)
print("Opinion:")
opinion_alpha = shapiro_wilk_test(fscore_opinion)
print("Fake News:")
fake_alpha = shapiro_wilk_test(fscore_fake)
print("Weighted Average:")
weighted_alpha = shapiro_wilk_test(weighted_avg)

print("\n")
print("Comparações de um algoritmo em relação aos demais algoritmos:")
print("Testes comparando com: 'RF + Oversampling'")
print("Classe NEWS")
print("Decision Tree:")
perform_fulltest(rf_ov_news, dt_news, rf_ov_news_alpha, dt_news_alpha)
print("Decision Tree + Oversampling:")
perform_fulltest(rf_ov_news, dt_ov_news, rf_ov_news_alpha, dt_ov_news_alpha)
print("Decision Tree + Oversampling + Undersampling:")
perform_fulltest(rf_ov_news, dt_ov_un_news, rf_ov_news_alpha, dt_ov_un_news_alpha)
print("Multinomial Naive Bayes:")
perform_fulltest(rf_ov_news, mnb_news, rf_ov_news_alpha, mnb_news_alpha)
print("MNB + Oversampling:")
perform_fulltest(rf_ov_news, mnb_ov_news, rf_ov_news_alpha, mnb_ov_news_alpha)
print("MNB + Oversampling + Undersampling:")
perform_fulltest(rf_ov_news, mnb_ov_un_news, rf_ov_news_alpha, mnb_ov_un_news_alpha)
print("SVC:")
perform_fulltest(rf_ov_news, svc_news, rf_ov_news_alpha, svc_news_alpha)
print("SVC + Oversampling:")
perform_fulltest(rf_ov_news, svc_ov_news, rf_ov_news_alpha, svc_ov_news_alpha)
print("SVC + Oversampling + Undersampling:")
perform_fulltest(rf_ov_news, svc_ov_un_news, rf_ov_news_alpha, svc_ov_un_news_alpha)
print("Random Forest:")
perform_fulltest(rf_ov_news, rf_news, rf_ov_news_alpha, rf_news_alpha)
print("Random Forest + Oversampling + Undersampling:")
perform_fulltest(rf_ov_news, rf_ov_un_news, rf_ov_news_alpha, rf_ov_un_news_alpha)
print("Adaboost:")
perform_fulltest(rf_ov_news, ada_news, rf_ov_news_alpha, ada_news_alpha)
print("Adaboost + Oversampling:")
perform_fulltest(rf_ov_news, ada_ov_news, rf_ov_news_alpha, ada_ov_news_alpha)
print("Adaboost + Oversampling + Undersampling:")
perform_fulltest(rf_ov_news, ada_ov_un_news, rf_ov_news_alpha, ada_ov_un_news_alpha)
print("XGBoost:")
perform_fulltest(rf_ov_news, xgb_news, rf_ov_news_alpha, xgb_news_alpha)
print("XGBoost + Oversampling:")
perform_fulltest(rf_ov_news, xgb_ov_news, rf_ov_news_alpha, xgb_ov_news_alpha)
print("XGBoost + Oversampling + Undersampling:")
perform_fulltest(rf_ov_news, xgb_ov_un_news, rf_ov_news_alpha, xgb_ov_un_news_alpha)

print("\n")
print("Comparações de um algoritmo em relação aos demais algoritmos:")
print("Testes comparando com: 'Random Forest + Oversampling + Undersampling'")
print("Classe OPINION")
print("Decision Tree:")
perform_fulltest(rf_ov_un_opinion, dt_opinion, rf_ov_un_opinion_alpha, dt_opinion_alpha)
print("Decision Tree + Oversampling:")
perform_fulltest(rf_ov_un_opinion, dt_ov_opinion, rf_ov_un_opinion_alpha, dt_ov_opinion_alpha)
print("Decision Tree + Oversampling + Undersampling:")
perform_fulltest(rf_ov_un_opinion, dt_ov_un_opinion, rf_ov_un_opinion_alpha, dt_ov_un_opinion_alpha)
print("Multinomial Naive Bayes:")
perform_fulltest(rf_ov_un_opinion, mnb_opinion, rf_ov_un_opinion_alpha, mnb_opinion_alpha)
print("MNB + Oversampling:")
perform_fulltest(rf_ov_un_opinion, mnb_ov_opinion, rf_ov_un_opinion_alpha, mnb_ov_opinion_alpha)
print("MNB + Oversampling + Undersampling:")
perform_fulltest(rf_ov_un_opinion, mnb_ov_un_opinion, rf_ov_un_opinion_alpha, mnb_ov_un_opinion_alpha)
print("SVC:")
perform_fulltest(rf_ov_un_opinion, svc_opinion, rf_ov_un_opinion_alpha, svc_opinion_alpha)
print("SVC + Oversampling:")
perform_fulltest(rf_ov_un_opinion, svc_ov_opinion, rf_ov_un_opinion_alpha, svc_ov_opinion_alpha)
print("SVC + Oversampling + Undersampling:")
perform_fulltest(rf_ov_un_opinion, svc_ov_un_opinion, rf_ov_un_opinion_alpha, svc_ov_un_opinion_alpha)
print("Random Forest:")
perform_fulltest(rf_ov_un_opinion, rf_opinion, rf_ov_un_opinion_alpha, rf_opinion_alpha)
print("Random Forest + Oversampling:")
perform_fulltest(rf_ov_un_opinion, rf_ov_opinion, rf_ov_un_opinion_alpha, rf_ov_opinion_alpha)
print("Adaboost:")
perform_fulltest(rf_ov_un_opinion, ada_opinion, rf_ov_un_opinion_alpha, ada_opinion_alpha)
print("Adaboost + Oversampling:")
perform_fulltest(rf_ov_un_opinion, ada_ov_opinion, rf_ov_un_opinion_alpha, ada_ov_opinion_alpha)
print("Adaboost + Oversampling + Undersampling:")
perform_fulltest(rf_ov_un_opinion, ada_ov_un_opinion, rf_ov_un_opinion_alpha, ada_ov_un_opinion_alpha)
print("XGBoost:")
perform_fulltest(rf_ov_un_opinion, xgb_opinion, rf_ov_un_opinion_alpha, xgb_opinion_alpha)
print("XGBoost + Oversampling:")
perform_fulltest(rf_ov_un_opinion, xgb_ov_opinion, rf_ov_un_opinion_alpha, xgb_ov_opinion_alpha)
print("XGBoost + Oversampling + Undersampling:")
perform_fulltest(rf_ov_un_opinion, xgb_ov_un_opinion, rf_ov_un_opinion_alpha, xgb_ov_un_opinion_alpha)

print("\n")
print("Comparações de um algoritmo em relação aos demais algoritmos:")
print("Testes comparando com: 'XGBoost + Oversampling'")
print("Classe FAKE")
print("Decision Tree:")
perform_fulltest(xgb_ov_fake, dt_fake, xgb_ov_fake_alpha, dt_fake_alpha)
print("Decision Tree + Oversampling:")
perform_fulltest(xgb_ov_fake, dt_ov_fake, xgb_ov_fake_alpha, dt_ov_fake_alpha)
print("Decision Tree + Oversampling + Undersampling:")
perform_fulltest(xgb_ov_fake, dt_ov_un_fake, xgb_ov_fake_alpha, dt_ov_un_fake_alpha)
print("MNB:")
perform_fulltest(xgb_ov_fake, mnb_fake, xgb_ov_fake_alpha, mnb_fake_alpha)
print("MNB + Oversampling:")
perform_fulltest(xgb_ov_fake, mnb_ov_fake, xgb_ov_fake_alpha, mnb_ov_fake_alpha)
print("MNB + Oversampling + Undersampling:")
perform_fulltest(xgb_ov_fake, mnb_ov_un_fake, xgb_ov_fake_alpha, mnb_ov_un_fake_alpha)
print("SVC:")
perform_fulltest(xgb_ov_fake, svc_fake, xgb_ov_fake_alpha, svc_fake_alpha)
print("SVC + Oversampling:")
perform_fulltest(xgb_ov_fake, svc_ov_fake, xgb_ov_fake_alpha, svc_ov_fake_alpha)
print("SVC + Oversampling + Undersampling:")
perform_fulltest(xgb_ov_fake, svc_ov_un_fake, xgb_ov_fake_alpha, svc_ov_un_fake_alpha)
print("Random Forest:")
perform_fulltest(xgb_ov_fake, rf_fake, xgb_ov_fake_alpha, rf_fake_alpha)
print("Random Forest + Oversampling:")
perform_fulltest(xgb_ov_fake, rf_ov_fake, xgb_ov_fake_alpha, rf_ov_fake_alpha)
print("Random Forest + Oversampling + Undersampling:")
perform_fulltest(xgb_ov_fake, rf_ov_un_fake, xgb_ov_fake_alpha, rf_ov_un_fake_alpha)
print("Adaboost:")
perform_fulltest(xgb_ov_fake, ada_fake, xgb_ov_fake_alpha, ada_fake_alpha)
print("Adaboost + Oversampling:")
perform_fulltest(xgb_ov_fake, ada_ov_fake, xgb_ov_fake_alpha, ada_ov_fake_alpha)
print("Adaboost + Oversampling + Undersampling:")
perform_fulltest(xgb_ov_fake, ada_ov_un_fake, xgb_ov_fake_alpha, ada_ov_un_fake_alpha)
print("XGBoost:")
perform_fulltest(xgb_ov_fake, xgb_fake, xgb_ov_fake_alpha, xgb_fake_alpha)
print("XGBoost + Oversampling + Undersampling:")
perform_fulltest(xgb_ov_fake, xgb_ov_un_fake, xgb_ov_fake_alpha, xgb_ov_un_fake_alpha)

end = timeit.default_timer()
print ('Duração: %f segundos' % (end - start))