import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data1 = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\\Databse_IIIV.xlsx")
data1 = data1.drop(data1.index[0])
data2 = data1.values
feature_names = data1.columns[:-2].tolist()
X = data2[:, :-2]
Y = data2[:, -2:]
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=70)
# 数据归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 定义元学习器
meta_learner = GradientBoostingRegressor(n_estimators=35, learning_rate=0.1, max_depth=5, max_features='sqrt', random_state=42)
# 定义基学习器配置
base_learners1 = [
    SVR(kernel='rbf', C=100, gamma=0.01),
    RandomForestRegressor(max_depth=10, n_estimators=30, random_state=42),
    MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=400)
]
base_learners2 = [
    SVR(kernel='rbf', C=100, gamma=0.5),
    RandomForestRegressor(max_depth=10, n_estimators=60, random_state=42),
    MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=600)
]
# 训练和评估堆叠模型
results = {}
target_names = ['Target 1', 'Target 2']
predictions = {name: [] for name in target_names}  # 存储每个目标的预测值

for i, base_learners in enumerate([base_learners1, base_learners2]):
    print(f"Training and evaluating model for {target_names[i]}")
    y_train_target = y_train[:, i]
    y_test_target = y_test[:, i]
    # 定义并训练堆叠模型
    stacked_regressor = StackingRegressor(regressors=base_learners, meta_regressor=meta_learner)
    stacked_regressor.fit(x_train, y_train_target)
    y_train_pred = stacked_regressor.predict(x_train)
    y_test_pred = stacked_regressor.predict(x_test)
    predictions[target_names[i]] = (y_train_pred, y_test_pred)  # 存储预测值
    # 计算并打印测试集的评估指标
    results[target_names[i]] = {
        'Test R2': r2_score(y_test_target, y_test_pred),
        'Test MSE': mean_squared_error(y_test_target, y_test_pred),
        'Test MAE': mean_absolute_error(y_test_target, y_test_pred)
    }
    print(f"{target_names[i]} - Test R2:", results[target_names[i]]['Test R2'])
    print(f"{target_names[i]} - Test MSE:", results[target_names[i]]['Test MSE'])
    print(f"{target_names[i]} - Test MAE:", results[target_names[i]]['Test MAE'])
    # 计算并打印训练集的评估指标
    results[target_names[i]].update({
        'Train R2': r2_score(y_train_target, y_train_pred),
        'Train MSE': mean_squared_error(y_train_target, y_train_pred),
        'Train MAE': mean_absolute_error(y_train_target, y_train_pred)
    })
    print(f"{target_names[i]} - Train R2:", results[target_names[i]]['Train R2'])
    print(f"{target_names[i]} - Train MSE:", results[target_names[i]]['Train MSE'])
    print(f"{target_names[i]} - Train MAE:", results[target_names[i]]['Train MAE'])
    print("---------------------finished----------------------")

# 带边际分布的散点密度图
sns.set(style="whitegrid")
colors = ['#2da7e2', '#a5aaa3']
def create_jointplot(y_true_train, y_pred_train, y_true_test, y_pred_test, target_name, colors):
    data = pd.DataFrame({
        'Actual': np.concatenate((y_true_train, y_true_test)),
        'Predicted': np.concatenate((y_pred_train, y_pred_test)),
        'Dataset': ['Train'] * len(y_true_train) + ['Test'] * len(y_true_test)
    })
    fontdict = {
        'fontname': 'Times New Roman',
        'fontsize': 20,
        'fontweight': 'bold',  # 或者使用 'heavy', 'light', 'normal', 'ultrabold' 等
        'color': 'black'  # 如果你还想设置字体颜色
    }
    font = {
        'family': 'Times New Roman',
        'size': 20,
        'weight': 'normal',  # 或者使用 'heavy', 'light', 'normal', 'ultrabold' 等
    }
    sns.jointplot(data=data, x='Actual', y='Predicted', kind='scatter', hue='Dataset',
                  palette={"Train": colors[0], "Test": colors[1]}, height=6, ratio=10)
    min_val = min(np.min(data['Actual']), np.min(data['Predicted']))
    max_val = max(np.max(data['Actual']), np.max(data['Predicted']))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.xlabel(f'DFT-calculated {target_name}', fontdict)
    plt.ylabel(f'Predicted {target_name}', fontdict)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.legend(loc='upper left',prop={'size': 18, 'family': 'Times New Roman'})
    plt.show()

create_jointplot(y_train[:, 0], predictions['Target 1'][0], y_test[:, 0], predictions['Target 1'][1], 'Adsorption Energy', colors)
create_jointplot(y_train[:, 1], predictions['Target 2'][0], y_test[:, 1], predictions['Target 2'][1], 'Work Function', colors)


