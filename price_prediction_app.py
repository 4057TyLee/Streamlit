
import pandas as pd   
import streamlit as st   
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.model_selection import cross_validate
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
import pickle
from PIL import Image

st.sidebar.title('Car Price Prediction')

html_temp = """
<div style="background-color:teal;padding:10px">
<h1 style="color:white;text-align:center;">Car Price Prediction </h1>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

# İmages
image = Image.open("cars.png")
st.image(image, use_column_width=True)

# To load machine learning model
filename = "model"
model=pickle.load(open(filename, "rb"))


# To take feature inputs
make_model = st.sidebar.selectbox("Make Model",['Audi A1', 
                                        'Audi A2', 
                                        'Audi A3', 
                                        'Opel Astra', 
                                        'Opel Corsa', 
                                        'Opel Insignia', 
                                        'Renault Clio', 
                                        'Renault Duster', 
                                        'Renault Espace'])

gearing_type = st.sidebar.selectbox('Gearing Type', ['Automatic', 'Manual', 'Semi-automatic'])

km = st.sidebar.number_input("Km:",min_value=0, max_value=317000)

age = st.sidebar.slider("Age:",min_value=0, max_value=3)

hp_kw = st.sidebar.slider("hp_kw:",min_value=40, max_value=294)

gears = st.sidebar.slider("Gears:", 5, 8)


# Create a dataframe using feature inputs
sample = {'make_model': make_model,
          'gearing_type': gearing_type,
          'age': age,
          'km': km,
          'hp_kw': hp_kw,
          'gears': gears
          }

df = pd.DataFrame.from_dict([sample])

st.header("The configuration of your car is below")

st.table(df)

st.subheader("Press predict if configuration is okay")

# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)
if predict :
    st.success("The estimated price of your car is {} €. ".format(int(result[0])))
    