#%%
import pandas as pd 
import numpy as np 
from matplotlib import pyplot
import seaborn as sns 
from sklearn import feature_selection as fs
# %%
diabetes_bin = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
diabetes_bin.head()
# %%
#feature selection 
#split data first and do feature selection solely on split
# create select fpr, fit transform with data 
stats, pvals = fs.chi2(diabetes_bin.drop('Diabetes_binary', axis = 1), diabetes_bin['Diabetes_binary'])
# %%
