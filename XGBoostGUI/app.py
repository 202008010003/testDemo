from flask import Flask, render_template, request, jsonify, redirect, url_for,send_from_directory
import numpy as np
import pickle  # 假设我们用pickle保存了训练好的模型
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
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
# 初始化Flask应用
app = Flask(__name__)

# 定义上传文件存储路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 提供访问上传文件的路由
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/train_model', methods=['POST'])
def train_model():
    # 获取上传的文件
    file = request.files['file']
    if not file:
        return "没有文件上传", 400

    # 获取文件扩展名并检查文件类型
    file_extension = file.filename.split('.')[-1].lower()

    # 只允许 CSV 和 Excel 文件
    if file_extension == 'csv':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  # 设置保存路径
        file.save(file_path)  # 保存文件
        data = pd.read_csv(file_path)  # 使用 pandas 读取文件
    elif file_extension in ['xls', 'xlsx']:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        data = pd.read_excel(file_path)  # 使用 pandas 读取 Excel 文件
    else:
        return "不支持的文件类型", 400  # 如果不是支持的文件类型，则返回错误

    # 特征和目标变量定义
    X = data[['x1', 'x2', 'x3', 'x4']]  # 输入特征
    y1 = data['y1']                     # 输出目标变量 y1
    y2 = data['y2']                     # 输出目标变量 y2

    # 将数据集分为训练集和测试集, random_state=42
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=7)
    _, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=7)

    # 定义y1参数范围
    param_dist_y1 = {
        'n_estimators': np.arange(50, 121, 5),  # 决策树的数量，从100到150，步长为10
        'max_depth': np.arange(1, 4,1),          # 每棵树的最大深度，从2到6
        'learning_rate': np.logspace(-2, -1, 10),  # 学习率在0.01到0.1之间的对数分布
        'subsample': np.linspace(0.6 ,0.9, 10),  # 样本比例从0.6到0.9
        'colsample_bytree': np.linspace(0.6, 1, 10),  # 特征采样比例
        'reg_alpha': np.logspace(-4, -3, 20),    # L1正则化系数，从1e-4到1e-2之间的对数分布
        'reg_lambda': np.logspace(-4, -3, 20),   # L2正则化系数，从1e-4到1e-2之间的对数分布
        'random_state': [42]                     # 固定随机种子
    }

    # 定义y2参数范围
    param_dist_y2 = {
        'n_estimators': np.arange(50, 151, 5),  # 决策树的数量，从100到150，步长为10
        'max_depth': np.arange(1, 4,1),          # 每棵树的最大深度，从2到6
        'learning_rate': np.logspace(-2, -1, 10),  # 学习率在0.01到0.1之间的对数分布
        'subsample': np.linspace(0.6, 1, 10),    # 样本比例从0.6到0.9
        'colsample_bytree': np.linspace(0.6, 1, 10),  # 特征采样比例
        'reg_alpha': np.logspace(-3, -1, 50),    # L1正则化系数，从1e-4到1e-2之间的对数分布
        'reg_lambda': np.logspace(-4, -3, 20),   # L2正则化系数，从1e-4到1e-2之间的对数分布
        'random_state': [42]                     # 固定随机种子
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

    best_params_y1, best_score_y1 = randomized_search_y1(X_train, y1_train)
    best_params_y2, best_score_y2 = randomized_search_y2(X_train, y2_train)

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

    # 绘制测试集拟合图：y1 和 y2
    plt.figure(figsize=(10, 6))

    # 绘制 y1 测试集拟合图
    plt.subplot(1, 2, 1)
    plt.scatter(y1_test, y1_pred, color='blue', label='y1: Actual vs Predicted')
    plt.plot([y1_test.min(), y1_test.max()], [y1_test.min(), y1_test.max()], color='red', lw=2)
    plt.title('y1: Test Set Fitting')
    plt.xlabel('Actual y1')
    plt.ylabel('Predicted y1')
    plt.legend()

    # 绘制 y2 测试集拟合图
    plt.subplot(1, 2, 2)
    plt.scatter(y2_test, y2_pred, color='green', label='y2: Actual vs Predicted')
    plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], color='red', lw=2)
    plt.title('y2: Test Set Fitting')
    plt.xlabel('Actual y2')
    plt.ylabel('Predicted y2')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig('uploads/model_fit.png')  # 保存图表图片
    plt.close()  # 关闭图像，释放内存

    # 保存模型
    joblib.dump(xgb_y1, 'xgboost_y1_model.pkl')
    joblib.dump(xgb_y2, 'xgboost_y2_model.pkl')

    # 将训练结果和测试结果传递给前端页面
    return redirect(url_for('train_result_page', mse_y1_train=mse_y1_train, rmse_y1_train=rmse_y1_train,
                            mae_y1_train=mae_y1_train, mape_y1_train=mape_y1_train, r2_y1_train=r2_y1_train,
                            mse_y2_train=mse_y2_train, rmse_y2_train=rmse_y2_train,
                            mae_y2_train=mae_y2_train, mape_y2_train=mape_y2_train, r2_y2_train=r2_y2_train,
                            mse_y1_test=mse_y1_test, rmse_y1_test=rmse_y1_test, mae_y1_test=mae_y1_test,
                            mape_y1_test=mape_y1_test, r2_y1_test=r2_y1_test,
                            mse_y2_test=mse_y2_test, rmse_y2_test=rmse_y2_test, mae_y2_test=mae_y2_test,
                            mape_y2_test=mape_y2_test, r2_y2_test=r2_y2_test))


@app.route('/train_result')
def train_result_page():
    # 从URL获取传递的参数
    mse_y1_train = request.args.get('mse_y1_train')
    rmse_y1_train = request.args.get('rmse_y1_train')
    mae_y1_train = request.args.get('mae_y1_train')
    mape_y1_train = request.args.get('mape_y1_train')
    r2_y1_train = request.args.get('r2_y1_train')

    mse_y1_test = request.args.get('mse_y1_test')
    rmse_y1_test = request.args.get('rmse_y1_test')
    mae_y1_test = request.args.get('mae_y1_test')
    mape_y1_test = request.args.get('mape_y1_test')
    r2_y1_test = request.args.get('r2_y1_test')

    mse_y2_train = request.args.get('mse_y2_train')
    rmse_y2_train = request.args.get('rmse_y2_train')
    mae_y2_train = request.args.get('mae_y2_train')
    mape_y2_train = request.args.get('mape_y2_train')
    r2_y2_train = request.args.get('r2_y2_train')

    mse_y2_test = request.args.get('mse_y2_test')
    rmse_y2_test = request.args.get('rmse_y2_test')
    mae_y2_test = request.args.get('mae_y2_test')
    mape_y2_test = request.args.get('mape_y2_test')
    r2_y2_test = request.args.get('r2_y2_test')

    # 将数据传递给模板进行渲染
    return render_template('train_result.html', mse_y1_train=mse_y1_train, rmse_y1_train=rmse_y1_train,
                           mae_y1_train=mae_y1_train, mape_y1_train=mape_y1_train, r2_y1_train=r2_y1_train,
                           mse_y2_train=mse_y2_train, rmse_y2_train=rmse_y2_train,
                           mae_y2_train=mae_y2_train, mape_y2_train=mape_y2_train, r2_y2_train=r2_y2_train,
                           mse_y1_test=mse_y1_test, rmse_y1_test=rmse_y1_test, mae_y1_test=mae_y1_test,
                           mape_y1_test=mape_y1_test, r2_y1_test=r2_y1_test,
                           mse_y2_test=mse_y2_test, rmse_y2_test=rmse_y2_test, mae_y2_test=mae_y2_test,
                           mape_y2_test=mape_y2_test, r2_y2_test=r2_y2_test)

# 用于暂时存储用户数据的字典
users = {'admin': '123'}

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # 检查用户名和密码是否匹配
    if username in users and users[username] == password:
        return redirect(url_for('model_page'))  # 登录成功后跳转到model.html页面
    else:
        return "用户名或密码错误，请重新登录。"

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']

    # 检查用户名是否已存在
    if username in users:
        return "用户名已存在，请选择其他用户名。"

    # 将用户信息存储到字典中
    users[username] = password
    return redirect(url_for('login_page'))  # 注册成功后跳转到登录页面


# 加载训练好的模型
# 这里假设模型是一个pickle文件，实际上可以加载任何训练好的模型
def load_y1_model():
    # 假设我们使用pickle保存了训练好的模型
    with open("xgboost_y1_model.pkl", "rb") as f:
        y1_model = pickle.load(f)
    return y1_model

def load_y2_model():
    # 假设我们使用pickle保存了训练好的模型
    with open("xgboost_y2_model.pkl", "rb") as f:
        y2_model = pickle.load(f)
    return y2_model


# 用于预测的函数
def predict(x1, x2, x3, x4):
    # 假设模型是一个回归模型，且接收一个四维特征向量
    print(f"Input Features: x1={x1}, x2={x2}, x3={x3}, x4={x4}")  # 打印输入特征
    model_y1 = load_y1_model()
    features = np.array([[x1, x2, x3, x4]])
    y1 = model_y1.predict(features)

    model_y2 = load_y2_model()
    y2 = model_y2.predict(features)

    return float(y1[0]), float(y2[0])

# # 首页路由
# @app.route('/')
# def index():
#     return render_template('predict.html')
@app.route('/predict_page')
def predict1():
    return render_template('predict.html')


# 预测请求路由
@app.route('/predict', methods=['POST'])
def predict_route():
    # 从前端接收数据
    try:
        x1 = float(request.form['x1'])
        x2 = float(request.form['x2'])
        x3 = float(request.form['x3'])
        x4 = float(request.form['x4'])

        # 调用预测函数
        y1, y2 = predict(x1, x2, x3, x4)
        # 打印预测结果
        print(f"Predictions - y1: {y1}, y2: {y2}")
        # 返回预测结果
        return jsonify({'y1': y1, 'y2': y2})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
