import numpy as np
import pandas as pd
from sklearn import preprocessing, svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from cleanData import *

df = pd.read_csv(".\\data\\testA.csv")
out = df[['ID','TOOL_ID']]
df = drop_fill(df)
df = read_Dictionary_replace(df)
delcols = np.loadtxt("delcols.txt",dtype=bytes).astype(str)
df.drop(delcols, 1, inplace=True)

X = np.array(df.values)
# print(X)
X = preprocessing.scale(X)


model = joblib.load("GBR_model.m")
y = model.predict(X)
# print(y[0:20])
out['y'] = y
print('out',out)
out.to_csv('024A_GBR.csv')