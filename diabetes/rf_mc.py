#%%
import pandas as pd 
import numpy as np 
from matplotlib import pyplot
import seaborn as sns 

import json 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, RocCurveDisplay, confusion_matrix 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFpr, chi2

import pickle 
# %%
diabetes_mc = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
diabetes_mc.head()
# %%
#feature selection 
#split data first and do feature selection solely on split
# create select fpr, fit transform with data 
X_train, X_test, y_train, y_test = train_test_split(diabetes_mc.iloc[:,1:], 
                                                    diabetes_mc['Diabetes_012'], 
                                                    test_size= 0.20, random_state= 777, 
                                                    shuffle = True, 
                                                    stratify= diabetes_mc['Diabetes_012'])
X_train = SelectFpr(chi2, alpha=0.01).fit_transform(X_train, y_train)
# %%
# Random Forest (random_state = 777, oob_score = True)
# grid search (n_jobs = 4, cv = 5, return_train_score = True, scoring = ['f1_weighted','precision_weighted','recall_weighted','roc_auc'], refit = 'f1_weighted')
params = {'n_estimators':np.arange(1000,4000,1000),
          'max_depth':[13,14,15,16,None],
          'max_features':['sqrt',0.5]}
clf_forest = RandomForestClassifier(random_state= 777,
                                    oob_score= True)
gcv = GridSearchCV(clf_forest, param_grid= params, 
                   n_jobs = 4, cv = 5, 
                   return_train_score = True, 
                   scoring = ['f1_weighted','precision_weighted',
                              'recall_weighted','roc_auc'], 
                    refit = 'f1_weighted').fit(X_train,y_train)
# f1 score
# %%
print(f'best validation F1 score: {gcv.cv_results_["mean_test_f1_weighted"][gcv.best_index_]}')
print(f'best training F1 score: {gcv.cv_results_["mean_train_f1_weighted"][gcv.best_index_]}')
print(gcv.best_params_)
# %%
with open('rf_baseline_results_mc.pkl','wb') as results:
    pickle.dump(gcv,results)

# %%
results = None
with open('rf_baseline_results_mc.pkl','rb') as results:
    results = pickle.load(results)

results

# %%
y_pred = results.best_estimator_.predict(X_train)
conf_mat = confusion_matrix(y_true = y_train, y_pred= y_pred)
conf_mat
# %%

# CHANGE NAMES FOR ALL VARS 
# gbt params
params = {'max_iter':np.arange(1000,4000,1000),
          'max_depth':[13,14,15,16,None],
          'max_leaf_nodes':[31,40,60],
          'learning_rate': [0.001,0.01,0.1]}
# HistGradientBoostingClassifier(random_state = 777)