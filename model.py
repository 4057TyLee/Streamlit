import pandas as pd      
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.model_selection import cross_validate



df= pd.read_csv('final_model.csv')

X = df.drop("price", axis=1)
y = df['price']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=101)

cat = X.select_dtypes("object").columns



ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', 
                         unknown_value=-1)

column_trans = make_column_transformer((ord_enc, cat), 
                                        remainder='passthrough',
                                        verbose_feature_names_out=False)

operations = [("OrdinalEncoder", column_trans), 
              ("XGB_model", XGBRegressor(n_estimators=45,
                                         learning_rate=0.3, 
                                         max_depth=5,
                                         subsample=1, 
                                         colsample_bylevel = 1,
                                         colsample_bytree = 0.8,
                                         random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X, y)


pickle.dump(pipe_model, open('model_xgb', 'wb'))