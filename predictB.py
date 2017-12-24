import numpy as np
import pandas as pd
from sklearn import preprocessing, svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.externals import joblib

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


df = pd.read_csv('testA.csv')
out = df[['ID','TOOL_ID']]
df.drop(['ID'], 1, inplace=True)
df.fillna(0, inplace=True)

df = handle_non_numerical_data(df)

X = np.array(df.values)
X = preprocessing.scale(X)
df.dropna(inplace=True)

model = joblib.load("DecisionTreeRegressor_model.m")
y = model.predict(X)
print(y[0:20])
out['y'] = y
out.to_csv('001A.csv')