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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFpr, chi2

import pickle 
# %%
diabetes_bin = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
diabetes_bin.head()
# %%
#feature selection 
#split data first and do feature selection solely on split
# create select fpr, fit transform with data 
X_train, X_test, y_train, y_test = train_test_split(diabetes_bin.iloc[:,1:], 
                                                    diabetes_bin['Diabetes_binary'], 
                                                    test_size= 0.20, random_state= 777, 
                                                    shuffle = True, stratify= diabetes_bin['Diabetes_binary'])
X_train = SelectFpr(chi2, alpha=0.01).fit_transform(X_train, y_train)
# %%
params = {'max_iter':np.arange(1000,4000,1000),
          'max_depth':[13,14,15,16,None],
          'min_samples_leaf':[40,60,80,100],
          'learning_rate': [0.001,0.01,0.1]}
clf_hgb = HistGradientBoostingClassifier(random_state= 777, categorical_features= [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
gcv = GridSearchCV(clf_hgb, param_grid= params, n_jobs = 4, 
                   cv = 5, return_train_score = True, 
                   scoring = ['f1_weighted','precision_weighted','recall_weighted','roc_auc'], 
                   refit = 'f1_weighted').fit(X_train,y_train)
# f1 score
# %%
print(f'best validation F1 score: {gcv.cv_results_["mean_test_f1_weighted"][gcv.best_index_]}')
print(f'best training F1 score: {gcv.cv_results_["mean_train_f1_weighted"][gcv.best_index_]}')
print(gcv.best_params_)
# %%
with open('hgb_baseline_results_bin.pkl','wb') as results:
    pickle.dump(gcv.cv_results_,results)

with open('hgb_baseline_model_bin.pkl','wb') as results:
    pickle.dump(gcv.best_estimator_,results)

# %%
results = None
with open('E:\work\diabetes_prediction\hgb_baseline_results_bin.pkl','rb') as results_file:
    results = pickle.load(results_file)

results

model = None
with open('E:\work\diabetes_prediction\hgb_baseline_model_bin.pkl','rb') as model_file:
    model = pickle.load(model_file)

# %%
y_pred = model.predict(X_train)
conf_mat = confusion_matrix(y_true = y_train, y_pred= y_pred)
conf_mat_df = pd.DataFrame(data = conf_mat, columns= ['Non-Diabetic','Diabetic'], index = ['Non-Diabetic','Diabetic'])

#%%
sns.heatmap(conf_mat_df, cbar= False, annot= True, fmt = 'd', cmap="crest")
# %%
np.bincount(diabetes_bin['Diabetes_binary'])
sns.barplot(x = [0.0,1.0], y = np.bincount(diabetes_bin['Diabetes_binary']))
# %%
