import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib

# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import ExtraTreesClassifier

toolCSV = {'TOOL_ID': 'TOOL_ID', 'TOOL_ID (#1)': 'TOOL_ID (#1)', 'TOOL_ID (#2)': 'TOOL_ID (#2)',
           'TOOL_ID (#3)': 'TOOL_ID (#3)',
           'TOOL': 'TOOL', 'TOOL (#1)': 'TOOL (#1)', 'TOOL (#2)': 'TOOL (#2)',
           'Tool': 'Tool_small', 'Tool (#2)': 'Tool (#2)_small', 'tool (#1)': 'tool (#1)_small',
           'Tool (#1)': 'Tool (#1)_num', 'tool': 'tool_num', 'Tool (#3)': 'Tool (#3)_num'}


def create_Tools_Dictionary(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if column in toolCSV:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            tempdf = pd.DataFrame(text_digit_vals, index=[0])
            tempdf.to_csv(".\\toolCSV\\%s.csv" % (toolCSV[column]))
            df[column] = list(map(convert_to_int, df[column]))
    return df


def read_Tools_Dictionary(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if column in toolCSV:
            tempdf = pd.read_csv(".\\toolCSV\\%s.csv" % (toolCSV[column]))
            text_digit_vals = tempdf.to_dict('index')[0]
            # print(text_digit_vals)
            df[column] = list(map(convert_to_int, df[column]))
    return df


def create_unique_Dictionary(unique_elements):
    x = 0
    text_digit_vals = {}
    for unique in unique_elements:
        if unique not in text_digit_vals:
            text_digit_vals[unique] = x
            x += 1
    return text_digit_vals


def create_NumClass_Dictionary(df):
    columns = df.columns.values
    numClass=[]
    for column in columns:
        if column not in toolCSV:
            unique_elements = sorted(set(df[column].astype(str)))
            if len(unique_elements) < 5:
                text_digit_vals = create_unique_Dictionary(unique_elements)
                df[column] = list(map(lambda x: text_digit_vals[x], df[column].astype(str)))
                tempdf = pd.DataFrame(text_digit_vals, index=[0])
                tempdf.to_csv(".\\col_class_CSV\\%s.csv" % column)
                numClass.append(column)
    return df, numClass


def read_NumClass_Dictionary(df):
    class_cols = np.loadtxt("class_cols.txt", dtype=bytes).astype(str)
    columns = df.columns.values
    for column in columns:
        if column in class_cols:
            tempdf = pd.read_csv(".\\col_class_CSV\\%s.csv" % column)
            text_digit_vals = tempdf.to_dict('index')[0]
            df[column] = list(map(lambda x: text_digit_vals[x], df[column].astype(str)))
    return df


def drop_fill(df):
    df.drop(['ID'], 1, inplace=True)
    df.replace(0, np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    df.fillna(0, inplace=True)  # 先用均值填充NaN,如果还有NaN说明该列全为零，用0填充，让后面删掉
    return df


def calculate_mse(_x, _y):
    return np.linalg.norm((_x - _y)) / len(_y)


def dropCharacter(df):
    # 通过方差选择特征。方差为0的特征会被自动移除。剩下的特征按设定的方差的阈值进行选择。
    stdcols = df.std().values.tolist()
    headers = df.columns.values
    zero_std_cols = []
    for index in range(len(stdcols)):
        if stdcols[index] <= 1e-14:
            zero_std_cols.append(headers[index])
    df.drop(zero_std_cols, 1, inplace=True)
    return df, zero_std_cols


def scale_train(x):
    # 范围0-1缩放标准化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # label_X_scaler=min_max_scaler.fit_transform(X)

    # lab_enc = preprocessing.LabelEncoder()
    # label_y = lab_enc.fit_transform(y)

    ##单变量特征选择-卡方检验，选择相关性最高的前100个特征  
    # X_chi2 = SelectKBest(chi2, k=2000).fit_transform(label_X_scaler, label_y)
    # print("训练集有 %d 行 %d 列" % (X_chi2.shape[0],X_chi2.shape[1]))
    # df_X_chi2=pd.DataFrame(X_chi2)
    # feature_names = df_X_chi2.columns.tolist()#显示列名
    # print('单变量选择的特征：\n',feature_names)
    ##通过方差选择特征。方差为0的特征会被自动移除。剩下的特征按设定的方差的阈值进行选择。  
    # sel = VarianceThreshold()#设置方差的阈值为超过80%都为同一个东西
    # print(label_X_scaler)
    # X_sel=sel.fit_transform(label_X_scaler)#选择方差大于0.6的特征
    # df_X_sel=pd.DataFrame(X_sel)
    # feature_names = df_X_sel.columns.tolist()#显示列名
    # print('方差选择的特征：\n',feature_names)
    # print(df_X_sel.head())

    ##基于L1的特征选择  
    ##lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(label_X_scaler, label_y)  
    ##model = SelectFromModel(lsvc, prefit=True)  
    ##X_lsvc = model.transform(label_X_scaler)  
    ##df_X_lsvc=pd.DataFrame(X_chi2)  
    ##feature_names = df_X_lsvc.columns.tolist()#显示列名  
    ##print('L1选择的特征：\n',feature_names)  

    ##基于树的特征选择，并按重要性阈值选择特征  
    # clf = ExtraTreesClassifier()#基于树模型进行模型选择
    # clf = clf.fit(label_X_scaler, label_y)
    # model = SelectFromModel(clf, threshold='1.00*mean',prefit=True)#选择特征重要性为1倍均值的特征，数值越高特征越重要
    # X_trees = model.transform(label_X_scaler)#返回所选的特征
    # df_X_trees=pd.DataFrame(X_chi2)
    # feature_names = df_X_trees.columns.tolist()#显示列名
    # print('树选择的特征：\n',feature_names)

    scaler = preprocessing.StandardScaler().fit(x)
    scaled_x = scaler.transform(x)
    joblib.dump(scaler, "scaler.save")
    return scaled_x


def scale_load(x):
    scaler = joblib.load("scaler.save")
    scaled_x = scaler.transform(x)
    return scaled_x


def findColumns(df, lower, upper):
    columns = df.columns.values
    colmean = df.mean()
    chosen_Col = []
    for column in columns:
        if (colmean[column] >= lower and colmean[column] <= upper):
            chosen_Col.append(column)
        else:
            pass
    # print("chosen cols:",chosen_Col)
    df.drop(chosen_Col, 1, inplace=True)
    return df, chosen_Col
