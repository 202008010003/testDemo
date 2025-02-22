import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import RandomizedSearchCV
import random
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.callback import EarlyStopping
from xgboost import XGBRegressor
import pickle
plt.rcParams['font.family']=['Times New Roman','SimHei']   #设置默认的中英文字体

# 读取数据
data = pd.read_excel('data0106_del.xlsx')

# 特征和目标变量定义
X = data[['x1', 'x2', 'x3', 'x4']]  # 输入特征
y1 = data['y1']                     # 输出目标变量 y1
y2 = data['y2']                     # 输出目标变量 y2

# 将数据集分为训练集和测试集, random_state=42
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=7)
_, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=7)

# 可视化目标变量在训练集和测试集中的分布
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei，以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
plt.figure(figsize=(10, 4))
plt.hist(y1_train, bins=20, alpha=0.7, label='训练集')
plt.hist(y1_test, bins=20, alpha=0.7, label='测试集')
plt.xlabel('y1')
plt.ylabel('频率')
plt.legend()
plt.title('训练集和测试集目标变量分布')
plt.show()

plt.figure(figsize=(10, 4))
plt.hist(y2_train, bins=20, alpha=0.7, label='训练集')
plt.hist(y2_test, bins=20, alpha=0.7, label='测试集')
plt.xlabel('y2')
plt.ylabel('频率')
plt.legend()
plt.title('训练集和测试集目标变量分布')
plt.show()

# 定义y1参数范围
param_dist_y1 = {
    'n_estimators': np.arange(50, 121, 5),  # 决策树的数量，从100到150，步长为10
    'max_depth': np.arange(1, 4,1),              # 每棵树的最大深度，从2到6
    'learning_rate': np.logspace(-2, -1, 10),  # 学习率在0.01到0.1之间的对数分布
    'subsample': np.linspace(0.6 ,0.9, 10),     # 样本比例从0.6到0.9
    'colsample_bytree': np.linspace(0.6, 1, 10),  # 特征采样比例
    'reg_alpha': np.logspace(-4, -3, 20),      # L1正则化系数，从1e-4到1e-2之间的对数分布
    'reg_lambda': np.logspace(-4, -3, 20),     # L2正则化系数，从1e-4到1e-2之间的对数分布
    'random_state': [42]                      # 固定随机种子
}

# 定义y2参数范围
param_dist_y2 = {
    'n_estimators': np.arange(50, 151, 5),  # 决策树的数量，从100到150，步长为10
    'max_depth': np.arange(1, 4,1),              # 每棵树的最大深度，从2到6
    'learning_rate': np.logspace(-2, -1, 10),  # 学习率在0.01到0.1之间的对数分布
    'subsample': np.linspace(0.6, 1, 10),     # 样本比例从0.6到0.9
    'colsample_bytree': np.linspace(0.6, 1, 10),  # 特征采样比例
    'reg_alpha': np.logspace(-3, -1, 50),      # L1正则化系数，从1e-4到1e-2之间的对数分布
    'reg_lambda': np.logspace(-4, -3, 20),     # L2正则化系数，从1e-4到1e-2之间的对数分布
    'random_state': [42]                      # 固定随机种子
}

# 定义并训练XGBRegressor模型，使用RandomizedSearchCV进行超参数搜索
def randomized_search_y1(X_train, y1_train):
    model = XGBRegressor()
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_dist_y1,
                                       n_iter=500,    # 进行500次随机搜索
                                       cv=5,         # 使用5折交叉验证
                                       scoring='neg_mean_squared_error',  # 使用负均方误差评分
                                       random_state=42,
                                       n_jobs=-1)    # 使用所有CPU核心
    random_search.fit(X_train, y1_train)
    return random_search.best_params_, random_search.best_score_

# 对y2进行随机搜索调参
def randomized_search_y2(X_train, y2_train):
    model = XGBRegressor()
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_dist_y2,
                                       n_iter=500,
                                       cv=5,
                                       scoring='neg_mean_squared_error',
                                       random_state=42,
                                       n_jobs=-1)
    random_search.fit(X_train, y2_train)
    return random_search.best_params_, random_search.best_score_

# 对y1进行随机搜索调参
best_params_y1, best_score_y1 = randomized_search_y1(X_train, y1_train)
print("Best parameters for y1:", best_params_y1)
print("Best score for y1:", best_score_y1)

print("————————————————————————————————————————————————————————————————————————————————————————————————")

# 对y2进行随机搜索调参
best_params_y2, best_score_y2 = randomized_search_y2(X_train, y2_train)
print("Best parameters for y2:", best_params_y2)
print("Best score for y2:", best_score_y2)

# 模型训练
xgb_y1 = XGBRegressor(**best_params_y1, early_stopping_rounds=10)
xgb_y2 = XGBRegressor(**best_params_y2, early_stopping_rounds=10)

xgb_y1.fit(X_train, y1_train, eval_set=[(X_train, y1_train), (X_test, y1_test)], verbose=True)
xgb_y2.fit(X_train, y2_train, eval_set=[(X_train, y2_train), (X_test, y2_test)], verbose=True)

# y1训练集性能评估
y1_train_pred = xgb_y1.predict(X_train)
y1_train_actual = y1_train

# 计算 y1 的训练集性能指标
mse_y1_train = mean_squared_error(y1_train_actual, y1_train_pred)
rmse_y1_train = np.sqrt(mse_y1_train)
mae_y1_train = mean_absolute_error(y1_train_actual, y1_train_pred)
mape_y1_train = np.mean(np.abs((y1_train_actual - y1_train_pred) / y1_train_actual)) * 100
r2_y1_train = r2_score(y1_train_actual, y1_train_pred)

# y1测试集性能评估
y1_pred = xgb_y1.predict(X_test)
y1_test_actual = y1_test

# 计算 y1 的测试集性能指标
mse_y1_test = mean_squared_error(y1_test_actual, y1_pred)
rmse_y1_test = np.sqrt(mse_y1_test)
mae_y1_test = mean_absolute_error(y1_test_actual, y1_pred)
mape_y1_test = np.mean(np.abs((y1_test_actual - y1_pred) / y1_test_actual)) * 100
r2_y1_test = r2_score(y1_test_actual, y1_pred)

# y2训练集性能评估
y2_train_pred = xgb_y2.predict(X_train)
y2_train_actual = y2_train

# 计算 y2 的训练集性能指标
mse_y2_train = mean_squared_error(y2_train_actual, y2_train_pred)
rmse_y2_train = np.sqrt(mse_y2_train)
mae_y2_train = mean_absolute_error(y2_train_actual, y2_train_pred)
mape_y2_train = np.mean(np.abs((y2_train_actual - y2_train_pred) / y2_train_actual)) * 100
r2_y2_train = r2_score(y2_train_actual, y2_train_pred)

# y2测试集性能评估
y2_pred = xgb_y2.predict(X_test)
y2_test_actual = y2_test

# 计算 y2 的测试集性能指标
mse_y2_test = mean_squared_error(y2_test_actual, y2_pred)
rmse_y2_test = np.sqrt(mse_y2_test)
mae_y2_test = mean_absolute_error(y2_test_actual, y2_pred)
mape_y2_test = np.mean(np.abs((y2_test_actual - y2_pred) / y2_test_actual)) * 100
r2_y2_test = r2_score(y2_test_actual, y2_pred)

# 打印y1和y2的性能指标
print(f"y1 - 训练集 MSE: {mse_y1_train}, RMSE: {rmse_y1_train}, MAE: {mae_y1_train}, MAPE: {mape_y1_train}%, R^2: {r2_y1_train}")
print(f"y1 - 测试集 MSE: {mse_y1_test}, RMSE: {rmse_y1_test}, MAE: {mae_y1_test}, MAPE: {mape_y1_test}%, R^2: {r2_y1_test}")
print(f"y2 - 训练集 MSE: {mse_y2_train}, RMSE: {rmse_y2_train}, MAE: {mae_y2_train}, MAPE: {mape_y2_train}%, R^2: {r2_y2_train}")
print(f"y2 - 测试集 MSE: {mse_y2_test}, RMSE: {rmse_y2_test}, MAE: {mae_y2_test}, MAPE: {mape_y2_test}%, R^2: {r2_y2_test}")

# 保存模型为 pickle 文件
with open("xgboost_y1_model.pkl", "wb") as model_file:
    pickle.dump(xgb_y1, model_file)
    print("xgboost_y1_model.pkl 保存成功！")  # 输出保存成功信息

with open("xgboost_y2_model.pkl", "wb") as model_file:
    pickle.dump(xgb_y2, model_file)
    print("xgboost_y2_model.pkl 保存成功！")  # 输出保存成功信息
