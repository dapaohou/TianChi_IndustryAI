import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ElasticNet  # 批量导入要实现的回归算法
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
from sklearn.externals import joblib

from cleanData import *


df = pd.read_csv(".\\data\\train.csv")
df = drop_fill(df)
df = read_Dictionary_replace(df)

df,date0 = findColumns(df,1e15,1e16)
df,date1 = findColumns(df,2e13,1e14)
df,date2 = findColumns(df,20170000,20180000)
df,zero_std_cols = dropCharacter(df)

"""
纪录删除掉的列
"""
date2.extend(zero_std_cols)
date1.extend(date2)
date0.extend(date1)
delcols = sorted(np.array(date0))
np.savetxt('delcols.txt', delcols, delimiter=',',fmt="%s")

X = np.array(df.drop(['Y'], 1))
y = np.array(df['Y'])
X = preprocessing.scale(X)


# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_tree =DecisionTreeRegressor(max_depth=4)  # 建立决策树模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['BayesianRidge', 'DecisionTree', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表
model_dic = [model_br, model_tree, model_etc, model_svr, model_gbr]  # 不同回归模型对象的集合

isTest = False
viewAll = False
if viewAll:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cv_score_list = []  # 交叉检验结果列表
    pre_y_list = []  # 各个回归模型预测的y值列表

    for model in model_dic:  # 读出每个回归模型对象
        scores = cross_val_score(model, X, y,scoring='neg_mean_squared_error',cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
        cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
        pre_y_list.append(model.fit(X_train, y_train).predict(X_test))  # 将回归训练中得到的预测y存入列表

    # 模型效果指标评估
    model_metrics_name = [mean_squared_error,explained_variance_score, mean_absolute_error, r2_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表
    for i in range(5):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y_test, pre_y_list[i])  # 计算每个回归指标结果
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

    df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['mse','ev', 'mae',  'r2'])  # 建立回归指标的数据框

    print('cross validation result:')  # 打印输出标题
    print(df1)  # 打印输出交叉检验的数据框
    print(70 * '-')  # 打印分隔线
    print('regression metrics:')  # 打印输出标题
    print(df2)  # 打印输出回归指标的数据框
    print(70 * '-')  # 打印分隔线
elif isTest:
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model_gbr.fit(X_train, y_train)
        MSE = calculate_mse(model_gbr.predict(X_test), y_test)
        print("MSE:", MSE)
else:
    model_gbr.fit(X,y)
    joblib.dump(model_gbr, "GBR_model.m")
# if isTest:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     model.fit(X_train,y_train)
#     y_predict = model.predict(X_test)
#     MSE = calculate_mse(y_predict,y_test)
#     print('MES:', MSE)
#
# else:
#     model.fit(X, y)
#     joblib.dump(model, "DecisionTreeRegressor_model.m")

# if isTest:
#     clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
#     clf.fit(X_train, y_train)
#     y_predict = model.predict(X_test)
#     MSE = calculate_mse(y_predict,y_test)
#     print('MES:', MSE)
# else:
#     clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
#     model.fit(X, y)
#     joblib.dump(clf, "SVR_model.m")
#
