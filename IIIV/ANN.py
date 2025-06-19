import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# 加载数据
data1 = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\\Databse_IIIV.xlsx")
data1 = data1.drop(data1.index[0])
data2 = data1.values
feature_names = data1.columns[:-2].tolist()
X = data2[:, :-2]
Y = data2[:, -2:]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=70)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 将归一化后的数据转换为DataFrame以便查看
x_train_df = pd.DataFrame(x_train, columns=feature_names)
x_test_df = pd.DataFrame(x_test, columns=feature_names)
print("归一化后的训练集前五行:")
print(x_train_df.head())
print("\n归一化后的测试集前五行:")
print(x_test_df.head())
print("---------------------finished----------------------")


#n_targets = y_train.shape[1]
#models = []
params_list = [
    {
    'hidden_layer_sizes': (100, 50),
    'activation': 'tanh',
    'max_iter': 400
    },
    {
    'hidden_layer_sizes': (100, 50),
    'activation':  'relu',
    'max_iter': 500
    }
]

# 初始化模型列表
models = []
y_train_preds = []
y_test_preds = []
losses = {
    'r2': [],
    'mean_squared_error': [],
    'mean_absolute_error': []
         }


for i in range(y_train.shape[1]):
    # 选择对应的超参数
    params = params_list[i]
    mlp_model = MLPRegressor(**params_list[i])
    mlp_model.fit(x_train, y_train[:, i])
    models.append(mlp_model)
    y_train_pred = mlp_model.predict(x_train)
    y_test_pred = mlp_model.predict(x_test)
    y_train_preds.append(y_train_pred)
    y_test_preds.append(y_test_pred)

    # 计算训练集和测试集的损失函数
    for dataset, y_true, y_pred, name in [
        (y_train, y_train[:, i], y_train_pred, "Train"),
        (y_test, y_test[:, i], y_test_pred, "Test")
    ]:
        score_r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        losses['r2'].append((score_r2, name))
        losses['mean_squared_error'].append((mse, name))
        losses['mean_absolute_error'].append((mae, name))

        print(f"Target {i + 1} - {name}:")
        print(f"R2: {score_r2}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        print("---------------------finished----------------------")


# 散点密度图
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
colors = ['blue', 'green']  # 颜色列表，用于区分不同的目标
for i, model in enumerate(models):
    plt.subplot(1, 2, i+1)
    plt.scatter(y_train[:, i], models[i].predict(x_train), alpha=0.7, c=colors[i], edgecolor='k', s=50, label='Train')
    plt.scatter(y_test[:, i], models[i].predict(x_test), alpha=0.7, c='red', edgecolor='k', s=50, label='Test')
    plt.plot([min(y_test[:, i]), max(y_test[:, i])],
             [min(y_test[:, i]), max(y_test[:, i])],
             color='yellow', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.xlabel('Actual Target Value', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Target Value', fontsize=14, fontweight='bold')
    plt.title(f'Actual vs Predicted Target {i+1} Values', fontsize=16)
    plt.legend()
plt.tight_layout()
plt.show()

'''plt.figure(figsize=(12, 6))  # 设置整个图形窗口的大小
plt.subplot(1, 2, 1)  # 1行2列的第一个子图
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7)
plt.plot([min(y_test[:, 0]), max(y_test[:, 0])],
         [min(y_test[:, 0]), max(y_test[:, 0])],
         color='red', linestyle='--')
plt.xlabel('Actual Target 1 Value')
plt.ylabel('Predicted Target 1 Value')
plt.title('Actual vs Predicted Target 1 Values')
# 第二个目标量的散点图和拟合线
plt.subplot(1, 2, 2)  # 1行2列的第二个子图
plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.7)
plt.plot([min(y_test[:, 1]), max(y_test[:, 1])],
         [min(y_test[:, 1]), max(y_test[:, 1])],
         color='red', linestyle='--')
plt.xlabel('Actual Target 2 Value')
plt.ylabel('Predicted Target 2 Value')
plt.title('Actual vs Predicted Target 2 Values')
plt.tight_layout()  # 自动调整子图布局以避免重叠
plt.show()'''

#回归预测图
X_test_index = np.arange(1, len(x_test) + 1, 1)
plt.figure(figsize=(12, 8))
for i in range(len(models)):
    ax = plt.subplot(2, 1, i + 1)  # 创建子图
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    plt.xticks(fontsize=20, fontweight='semibold')
    plt.yticks(fontsize=20, fontweight='semibold')
    plt.plot(X_test_index, y_test[:, i], marker='*', linestyle='-', color='#B54764', label="True", linewidth=1.5, markersize=8)
    plt.plot(X_test_index, y_test_preds[i], marker='o', linestyle='-', color='#8074C8', markeredgecolor="#8074C8", markerfacecolor=(0, 0, 0, 0), label="Prediction", linewidth=1.5, markersize=8)
    plt.xlabel("Sample Numbers", fontweight='semibold', fontname='Times New Roman', fontsize=28)
    plt.ylabel(f"Target {i+1}", fontweight='semibold', fontname='Times New Roman', fontsize=28)
    plt.title(f"RF for Target {i+1}", fontweight='semibold', fontname='Times New Roman', fontsize=28)
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontweight("semibold")
        text.set_fontname("Times New Roman")
        text.set_fontsize("18")
plt.tight_layout()
plt.show()


# 带边际分布的散点密度图
sns.set(style="whitegrid")
colors = ['#2da7e2', '#a5aaa3']
# 第一个目标
data1 = pd.DataFrame({
    'Actual': np.concatenate((y_train[:, 0], y_test[:, 0])),
    'Predicted': np.concatenate((models[0].predict(x_train), models[0].predict(x_test))),
    'Dataset': ['Train'] * len(y_train) + ['Test'] * len(y_test)
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
sns.jointplot(data=data1, x='Actual', y='Predicted', kind='scatter', hue='Dataset',
              palette={"Train": colors[0], "Test": colors[1]}, height=6, ratio=10)
min_val = min(np.min(data1['Actual']), np.min(data1['Predicted']))
max_val = max(np.max(data1['Actual']), np.max(data1['Predicted']))
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('DFT-calculated Eb', fontdict)
plt.ylabel('Predicted Eb', fontdict)
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
#plt.title('Actual vs Predicted Target 1 Values', fontsize=16)
plt.legend(loc='upper left',prop={'size': 18, 'family': 'Times New Roman'})
plt.show()
# 第二个目标
data2 = pd.DataFrame({
    'Actual': np.concatenate((y_train[:, 1], y_test[:, 1])),
    'Predicted': np.concatenate((models[1].predict(x_train), models[1].predict(x_test))),
    'Dataset': ['Train'] * len(y_train) + ['Test'] * len(y_test)
})
sns.jointplot(data=data2, x='Actual', y='Predicted', kind='scatter', hue='Dataset',
              palette={"Train": colors[0], "Test": colors[1]}, height=6, ratio=10)
min_val = min(np.min(data2['Actual']), np.min(data2['Predicted']))
max_val = max(np.max(data2['Actual']), np.max(data2['Predicted']))
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('DFT-calculated Eg', fontdict)
plt.ylabel('Predicted Eg', fontdict)
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
#plt.title('Actual vs Predicted Target 2 Values', fontsize=16)
plt.legend(loc='upper left',prop={'size': 18, 'family': 'Times New Roman'})
plt.show()
'''X = np.arange(1, len(x_test) + 1, 1)
ax = plt.gca()
ax.spines["top"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
plt.xticks(fontsize=20,fontweight='semibold')
plt.yticks(fontsize=20,fontweight='semibold')
plt.plot(X, y_test, marker='*', linestyle='-', color='#B54764', label="True", linewidth=1.5, markersize=8)
plt.plot(X, y_pred, marker='o', linestyle='-', color='#8074C8', markeredgecolor="#8074C8", markerfacecolor=(0, 0, 0, 0), label="prediction", linewidth=1.5, markersize=8)
plt.xlabel("Sample Numbers", fontweight='semibold',fontname='Times New Roman', fontsize=28)
plt.ylabel("Target", fontweight='semibold',fontname='Times New Roman', fontsize=28)
plt.title("ANN", fontweight='semibold',fontname='Times New Roman', fontsize=28)
legend=plt.legend()
legend.get_texts()[0].set_fontweight("semibold")
legend.get_texts()[0].set_fontname("Times New Roman")
legend.get_texts()[0].set_fontsize("18")
legend.get_texts()[1].set_fontweight("semibold")
legend.get_texts()[1].set_fontname("Times New Roman")
legend.get_texts()[1].set_fontsize("18")
plt.show()'''


'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


data1 = pd.read_excel("E:\excel-data\\alkali_metal_adsorption.xlsx")
data1 = data1.drop(data1.index[0])
data2 = data1.values
X = data2[:, :-2]
Y = data2[:, -2]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=70)
print(y_test)
print("---------------------finished----------------------")

#训练模型
nn_model = MLPRegressor(hidden_layer_sizes=(100,100), activation='tanh', max_iter=200)#多层感知器回归器
nn_model.fit(x_train, y_train)
y_pred = nn_model.predict(x_test)
print(y_pred)
X = np.arange(1, len(x_test) + 1, 1)

#损失函数
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
score1 = r2_score(y_test,y_pred)
print("R2 : ",score1)
print("均方误差 : ",mean_squared_error(y_test,y_pred))
print("均方误差开方 : ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("平均绝对误差 : ", mean_absolute_error(y_test, y_pred))
print("---------------------finished----------------------")

#散点密度图
xy = np.vstack([y_test, y_pred])  # 将两个维度的数据叠加
from scipy.stats import gaussian_kde
z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
idx = z.argsort()
x, y, z = y_test[idx], y_pred[idx], z[idx]
fig = plt.subplots()
ax = plt.gca()
ax.spines["top"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
plt.scatter(x, y, c=z, s=20, cmap='Spectral_r')  # c表示标记的颜色
plt.xlabel("True Value", fontweight='medium',fontname='Times New Roman', fontsize=36)
plt.ylabel("Predicted Value", fontweight='medium',fontname='Times New Roman', fontsize=36)
plt.xticks(fontsize=36, fontweight='medium')
plt.yticks(fontsize=36, fontweight='medium')
plt.title("MLP", fontweight='medium',fontname='Times New Roman', fontsize=36)
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(family='Times New Roman', size=36, weight='medium')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=36)  # 设置刻度标签大小
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font_prop)
plt.subplots_adjust(bottom=0.15)
plt.show()

X = np.arange(1, len(x_test) + 1, 1)
ax = plt.gca()
ax.spines["top"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
plt.xticks(fontsize=20,fontweight='semibold')
plt.yticks(fontsize=20,fontweight='semibold')
plt.plot(X, y_test, marker='*', linestyle='-', color='#B54764', label="True", linewidth=1.5, markersize=8)
plt.plot(X, y_pred, marker='o', linestyle='-', color='#8074C8', markeredgecolor="#8074C8", markerfacecolor=(0, 0, 0, 0), label="prediction", linewidth=1.5, markersize=8)
plt.xlabel("Sample Numbers", fontweight='semibold',fontname='Times New Roman', fontsize=28)
plt.ylabel("Target", fontweight='semibold',fontname='Times New Roman', fontsize=28)
plt.title("XGBoost", fontweight='semibold',fontname='Times New Roman', fontsize=28)
legend=plt.legend()
legend.get_texts()[0].set_fontweight("semibold")
legend.get_texts()[0].set_fontname("Times New Roman")
legend.get_texts()[0].set_fontsize("18")
legend.get_texts()[1].set_fontweight("semibold")
legend.get_texts()[1].set_fontname("Times New Roman")
legend.get_texts()[1].set_fontsize("18")
plt.show()'''

'''from sklearn.inspection import permutation_importance
# 计算排列特征重要性
perm_importance = permutation_importance(nn_model, x_test, y_test)
# 获取重要性数据并排序
sorted_idx = perm_importance.importances_mean.argsort()
for i in sorted_idx[::-1]:  # 按重要性降序排列
    print(f"Feature: {data1.columns[i]}, Importance: {perm_importance.importances_mean[i]}")
# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [data1.columns[i] for i in sorted_idx])
plt.title('Feature Importance')
plt.xlabel('Permutation Importance')
plt.show()'''


# 读取新的Excel文件
new_data = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict.xlsx")
x_new = new_data.iloc[:,:].values
y_pred_new = nn_model.predict(x_new)
new_data['Predicted_Target'] = y_pred_new
new_data.to_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict_result.xlsx", index=False)#生成的数据保存到目标路径

#绘制预测图
'''ax = plt.gca()
ax.spines["top"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
plt.xticks(fontsize=20,fontweight='semibold')
plt.yticks(fontsize=20,fontweight='semibold')
plt.plot(X, y_test, marker='*', linestyle='-', color='#B54764', label="True", linewidth=1.5, markersize=8)
plt.plot(X, y_pred, marker='o', linestyle='-', color='#8074C8', markeredgecolor="#8074C8", markerfacecolor=(0, 0, 0, 0), label="prediction", linewidth=1.5, markersize=8)
plt.xlabel("Sample Numbers", fontweight='semibold',fontname='Times New Roman', fontsize=28)
plt.ylabel("Absorptance(%)", fontweight='semibold',fontname='Times New Roman', fontsize=28)
plt.title("Neural Network Regression", fontweight='semibold',fontname='Times New Roman', fontsize=28)
legend=plt.legend()
legend.get_texts()[0].set_fontweight("semibold")
legend.get_texts()[0].set_fontname("Times New Roman")
legend.get_texts()[0].set_fontsize("18")
legend.get_texts()[1].set_fontweight("semibold")
legend.get_texts()[1].set_fontname("Times New Roman")
legend.get_texts()[1].set_fontsize("18")
plt.show()'''

'''feature_dict = {}
feature_names = ['D', 'H', 'S', 'W', 'Incidence', 'Component', 'Wavelength']
for feature_name in feature_names:
    value = input("请输入{}的值：".format(feature_name))
    feature_dict[feature_name] = float(value.strip())
# 构建特征向量
user_feature_vector = [feature_dict['D'], feature_dict['H'], feature_dict['S'], feature_dict['W'],
                      feature_dict['Incidence'], feature_dict['Component'], feature_dict['Wavelength']]
user_feature_vector = np.array(user_feature_vector).reshape(1, -1)
# 使用模型进行预测
predicted_target = nn_model.predict(user_feature_vector)
print("预测的目标值为:", predicted_target)'''

