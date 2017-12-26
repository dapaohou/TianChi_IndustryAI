from cleanData import *

df = pd.read_csv(".\\data\\testA.csv")
out = df[['ID', 'TOOL_ID']]
df = drop_fill(df)
df = read_Dictionary_replace(df)
delcols = np.loadtxt("delcols.txt", dtype=bytes).astype(str)
df.drop(delcols, 1, inplace=True)

X = np.array(df.values)
X = scale_load(X)
print(X.shape)

model = joblib.load("GBR_model.m")
y = model.predict(X)
# print(y[0:20])
out['y'] = y
print('out', out)
out.to_csv('026A_GBR.csv')
