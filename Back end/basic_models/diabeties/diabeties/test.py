import pandas as pd 
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import shap
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import pickle

# Assuming `model` is your trained XGBoost model
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
