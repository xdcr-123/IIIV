import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import palettable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
from sklearn.inspection import PartialDependenceDisplay
import shap
import lightgbm as lgb

# 读取数据
data1 = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\Databse_IIIV.xlsx")
data1 = data1.drop(data1.index[0])
data2 = data1.values
feature_names = data1.columns[:-2].tolist()
X = data2[:, :-2]
Y_1 = data2[:, -2]
Y_2 = data2[:, -1]
scaler = MinMaxScaler()#归一化
X = scaler.fit_transform(X)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X, Y_1, test_size=0.2, random_state=70)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X, Y_2, test_size=0.2, random_state=70)
# 将归一化后的数据转换为DataFrame以便查看
x_train_df_1 = pd.DataFrame(x_train_1, columns=feature_names)
x_test_df_1 = pd.DataFrame(x_test_1, columns=feature_names)
x_train_df_2 = pd.DataFrame(x_train_2, columns=feature_names)
x_test_df_2 = pd.DataFrame(x_test_2, columns=feature_names)
print("归一化后的训练集前五行:")
print(x_train_df_1.head(),x_train_df_2.head())
print("\n归一化后的测试集前五行:")
print(x_test_df_1.head(),x_test_df_2.head())

# 转换数据类型成为lightgbm数据类型
train_data_1 = lgb.Dataset(x_train_1, label=y_train_1)  # 使用 ravel() 将 y_train 转换为一维数组
test_data_1 = lgb.Dataset(x_test_1, label=y_test_1)
train_data_2 = lgb.Dataset(x_train_2, label=y_train_2)  # 使用 ravel() 将 y_train 转换为一维数组
test_data_2 = lgb.Dataset(x_test_2, label=y_test_2)

#目标1的训练机
# 设置参数
params_1 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 31,  #与max_depth正相关，这里相当树的深度为5
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 20,
    'verbose': 0
         }
# 训练模型
num_round = 100
bst = lgb.train(params_1, train_data_1, num_round, valid_sets=[test_data_1])
# 预测训练集和测试集
y_train_pred_1 = bst.predict(x_train_1, num_iteration=bst.best_iteration)
y_test_pred_1 = bst.predict(x_test_1, num_iteration=bst.best_iteration)
# 计算训练集的损失
train_r2 = r2_score(y_train_1, y_train_pred_1)
train_mse = mean_squared_error(y_train_1, y_train_pred_1)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_1, y_train_pred_1)
# 打印训练集的损失
print("Training Set Loss:")
print(f"R2 Score: {train_r2}")
print(f"Mean Squared Error: {train_mse}")
print(f"Root Mean Squared Error: {train_rmse}")
print(f"Mean Absolute Error: {train_mae}")
# 计算测试集的损失
test_r2 = r2_score(y_test_1, y_test_pred_1)
test_mse = mean_squared_error(y_test_1, y_test_pred_1)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_1, y_test_pred_1)
# 打印测试集的损失
print("Test Set Loss:")
print(f"R2 Score: {test_r2}")
print(f"Mean Squared Error: {test_mse}")
print(f"Root Mean Squared Error: {test_rmse}")
print(f"Mean Absolute Error: {test_mae}")

# 创建数据框用于绘图
data_1 = pd.DataFrame({
    'Actual': np.concatenate((y_train_1, y_test_1)),
    'Predicted': np.concatenate((y_train_pred_1, y_test_pred_1)),
    'Dataset': ['Train'] * len(y_train_1) + ['Test'] * len(y_test_1)
})

# 绘制带边际分布的散点密度图
sns.set(style="whitegrid")
colors = ['#2da7e2', '#a5aaa3']
fontdict = {
    'fontname': 'Times New Roman',
    'fontsize': 20,
    'fontweight': 'bold',
    'color': 'black'
}
font = {
    'family': 'Times New Roman',
    'size': 20,
    'weight': 'normal'
}
sns.jointplot(data=data_1, x='Actual', y='Predicted', kind='scatter', hue='Dataset',
              palette={"Train": colors[0], "Test": colors[1]}, height=6, ratio=10)
min_val = min(np.min(data_1['Actual']), np.min(data_1['Predicted']))
max_val = max(np.max(data_1['Actual']), np.max(data_1['Predicted']))
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual Adsorption Energy', fontdict)
plt.ylabel('Predicted Adsorption Energy', fontdict)
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
plt.legend(loc='upper left', prop={'size': 18, 'family': 'Times New Roman'})
plt.show()

'''plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_1, y=y_test_pred_1, alpha=0.5)
sns.kdeplot(x=y_test_1, y=y_test_pred_1, cmap='coolwarm', alpha=0.5, levels=20)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.show()'''


#目标2的训练机
params_2 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 31,  #与max_depth正相关，这里相当树的深度为5
    'learning_rate': 0.6,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 20,
    'verbose': 0
         }
# 训练模型
num_round = 100
bst = lgb.train(params_2, train_data_2, num_round, valid_sets=[test_data_2])
# 预测训练集和测试集
y_train_pred_2 = bst.predict(x_train_2, num_iteration=bst.best_iteration)
y_test_pred_2 = bst.predict(x_test_2, num_iteration=bst.best_iteration)
# 计算训练集的损失
train_r2 = r2_score(y_train_2, y_train_pred_2)
train_mse = mean_squared_error(y_train_2, y_train_pred_2)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_2, y_train_pred_2)
# 打印训练集的损失
print("Training Set Loss:")
print(f"R2 Score: {train_r2}")
print(f"Mean Squared Error: {train_mse}")
print(f"Root Mean Squared Error: {train_rmse}")
print(f"Mean Absolute Error: {train_mae}")
# 计算测试集的损失
test_r2 = r2_score(y_test_2, y_test_pred_2)
test_mse = mean_squared_error(y_test_2, y_test_pred_2)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_2, y_test_pred_2)
# 打印测试集的损失
print("Test Set Loss:")
print(f"R2 Score: {test_r2}")
print(f"Mean Squared Error: {test_mse}")
print(f"Root Mean Squared Error: {test_rmse}")
print(f"Mean Absolute Error: {test_mae}")

'''plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_2, y=y_test_pred_2, alpha=0.5)
sns.kdeplot(x=y_test_2, y=y_test_pred_2, cmap='coolwarm', alpha=0.5, levels=20)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.show()'''

# 创建数据框用于绘图
data_2 = pd.DataFrame({
    'Actual': np.concatenate((y_train_2, y_test_2)),
    'Predicted': np.concatenate((y_train_pred_2, y_test_pred_2)),
    'Dataset': ['Train'] * len(y_train_2) + ['Test'] * len(y_test_2)
})

# 绘制带边际分布的散点密度图
sns.set(style="whitegrid")
colors = ['#2da7e2', '#a5aaa3']
fontdict = {
    'fontname': 'Times New Roman',
    'fontsize': 20,
    'fontweight': 'bold',
    'color': 'black'
}
font = {
    'family': 'Times New Roman',
    'size': 20,
    'weight': 'normal'
}
sns.jointplot(data=data_2, x='Actual', y='Predicted', kind='scatter', hue='Dataset',
              palette={"Train": colors[0], "Test": colors[1]}, height=6, ratio=10)
min_val = min(np.min(data_2['Actual']), np.min(data_2['Predicted']))
max_val = max(np.max(data_2['Actual']), np.max(data_2['Predicted']))
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual Adsorption Energy', fontdict)
plt.ylabel('Predicted Adsorption Energy', fontdict)
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
plt.legend(loc='upper left', prop={'size': 18, 'family': 'Times New Roman'})
plt.show()

#预测数据
new_data = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict.xlsx")
x_val = new_data.iloc[:,:].values
dtest = lgb.DMatrix(x_val)
y_pred_ads = models[0].predict(dtest)
y_pred_wf = models[1].predict(dtest)
new_data['Predicted_adsorption'] = y_pred_ads
new_data['Predicted_work_function'] = y_pred_wf
new_data.to_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict_result.xlsx", index=False)
