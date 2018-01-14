from cleanData import *

df = pd.read_csv(".\\data\\testA.csv")
out = df[['ID']]
df = drop_fill(df)
df = read_Tools_Dictionary(df)
delcols = np.loadtxt(".\\data\\delcols.txt", dtype=bytes).astype(str)
df.drop(delcols, 1, inplace=True)

X = np.array(df.values)
print(X.shape)
X = scale_load(X)

model = joblib.load(".\\model\\DecisionTree.m")
y = model.predict(X)
# print(y[0:20])
out['y'] = y
print('out', out)
out.to_csv('.\\result\\114A_GBR.csv')
