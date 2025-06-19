import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

data1 = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\\Databse_IIIV_original.xlsx")
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


n_targets = y_train.shape[1]
models = []
# 为每个目标变量定义不同的超参数
params_list = [
    {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'eta': 0.2,
        'subsample': 1,
        'colsample_bytree': 0.5,
        'seed': 42
    },
    {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'seed': 43
    }
]

# 训练模型并计算损失
models = []
y_train_preds = []
y_test_preds = []
losses = {
    'r2': [],
    'mean_squared_error': [],
    'mean_absolute_error': []
}

for i in range(y_train.shape[1]):
    dtrain = xgb.DMatrix(x_train, label=y_train[:, i])
    dtest = xgb.DMatrix(x_test, label=y_test[:, i])
    params = params_list[i]
    model = xgb.train(params, dtrain, num_boost_round=100)
    models.append(model)

    # 预测训练集和测试集
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    y_train_preds.append(y_train_pred)
    y_test_preds.append(y_test_pred)
    # 计算训练集和测试集的损失函数
    for y_true, y_pred, name in [(y_train[:, i], y_train_pred, "Train"), (y_test[:, i], y_test_pred, "Test")]:
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        losses['r2'].append((r2, name))
        losses['mean_squared_error'].append((mse, name))
        losses['mean_absolute_error'].append((mae, name))

        print(f"Target {i + 1} - {name}:")
        print(f"R2: {r2}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {np.sqrt(mse)}")
        print(f"Mean Absolute Error: {mae}")
        print("---------------------finished----------------------")


# 绘制带边际分布的散点密度图
sns.set(style="whitegrid")
colors = ['#2da7e2', '#a5aaa3']
# 第一个目标
data1 = pd.DataFrame({
    'Actual': np.concatenate((y_train[:, 0], y_test[:, 0])),
    'Predicted': np.concatenate((models[0].predict(dtrain), models[0].predict(dtest))),
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
    'Predicted': np.concatenate((models[1].predict(dtrain), models[1].predict(dtest))),
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

#不带边际分布的散点图
'''sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
colors = ['blue', 'green']  # 颜色列表，用于区分不同的目标
for i, model in enumerate(models):
    dtrain = xgb.DMatrix(x_train, label=y_train[:, i])
    dtest = xgb.DMatrix(x_test, label=y_test[:, i])
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    plt.subplot(1, 2, i + 1)
    plt.scatter(y_train[:, i], y_train_pred, alpha=0.7, c=colors[i], edgecolor='k', s=50, label='Train')
    plt.scatter(y_test[:, i], y_test_pred, alpha=0.7, c='red', edgecolor='k', s=50, label='Test')
    plt.plot([min(y_test[:, i]), max(y_test[:, i])],
             [min(y_test[:, i]), max(y_test[:, i])],
             color='yellow', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.xlabel('Actual Target Value', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Target Value', fontsize=14, fontweight='bold')
    plt.title(f'Actual vs Predicted Target {i + 1} Values', fontsize=16)
    plt.legend()
plt.tight_layout()
plt.show()'''

#特征重要性
#feature_names = data1.columns[:-2]  # 假设最后两列是目标变量
# 获取第一个模型的特征重要性
sns.set(style="whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# 获取并绘制第一个模型的特征重要性
importance_1 = models[0].get_score(importance_type='weight')
sorted_importance_1 = sorted(importance_1.items(), key=lambda item: item[1], reverse=True)
features_1 = [feature_names[int(f[0][1:])] for f in sorted_importance_1]
importances_1 = [f[1] for f in sorted_importance_1]


# 使用渐变色绘制第一个模型的特征重要性
colors_1 = plt.cm.cividis_r(np.linspace(0, 1, len(importances_1)))
axs[0].bar(features_1, importances_1, color=colors_1)
axs[0].set_title('Feature Importance for Eb', fontsize=20, fontweight='bold')
axs[0].set_xlabel('Features', fontsize=30)
axs[0].set_ylabel('Importance', fontsize=30)
axs[0].set_xticklabels(features_1, rotation=90, fontsize=20)
axs[0].set_yticklabels(importances_1, fontsize=20)

# 获取并绘制第二个模型的特征重要性
importance_2 = models[1].get_score(importance_type='weight')
sorted_importance_2 = sorted(importance_2.items(), key=lambda item: item[1], reverse=True)
features_2 = [feature_names[int(f[0][1:])] for f in sorted_importance_2]
importances_2 = [f[1] for f in sorted_importance_2]
# 使用渐变色绘制第二个模型的特征重要性
colors_2 = plt.cm.Blues_r(np.linspace(0, 1, len(importances_2)))
axs[1].bar(features_2, importances_2, color=colors_2)
axs[1].set_title('Feature Importance for Eg', fontsize=20, fontweight='bold')
axs[1].set_xlabel('Features', fontsize=30)
axs[1].set_ylabel('Importance', fontsize=30)
axs[1].set_xticklabels(features_2, rotation=90, fontsize=20)
axs[1].set_yticklabels(importances_2, fontsize=20)
plt.tight_layout()
plt.show()
df_importance_1 = pd.DataFrame({'Feature': features_1, 'Importance': importances_1})
df_importance_2 = pd.DataFrame({'Feature': features_2, 'Importance': importances_2})
output_file_path1 = "C:\\Users\Dell\Desktop\IIIV\\feature_importance1.csv"
output_file_path2 = "C:\\Users\Dell\Desktop\IIIV\\feature_importance2.csv"
df_importance_1.to_csv(output_file_path1, index=False, mode='w')
df_importance_2.to_csv(output_file_path2, index=False, mode='a', header=False)
print("Feature importance for Target 1:")
print(df_importance_1)
print("\nFeature importance for Target 2:")
print(df_importance_2)


'''sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
xgb.plot_importance(models[0], ax=plt.gca(), importance_type='weight', max_num_features=19,
                    color='#1f77b4')  # 使用深蓝色
plt.title('Feature Importance for Target 1', fontsize=20, fontweight='bold')
plt.xlabel('Importance', fontsize=20)
plt.ylabel('Features', fontsize=20)
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
xgb.plot_importance(models[1], ax=plt.gca(), importance_type='weight', max_num_features=10,
                    color='#1f77b4')
plt.title('Feature Importance for Target 2', fontsize=20, fontweight='bold')
plt.xlabel('Importance', fontsize=20)
plt.ylabel('Features', fontsize=20)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''

'''
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
'''
#预测数据
new_data = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict.xlsx")
x_val = new_data.iloc[:,:].values
dtest = xgb.DMatrix(x_val)
y_pred_ads = models[0].predict(dtest)
y_pred_wf = models[1].predict(dtest)
new_data['Predicted_adsorption'] = y_pred_ads
new_data['Predicted_work_function'] = y_pred_wf
new_data.to_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict_result.xlsx", index=False)



















'''from sklearn.model_selection import GridSearchCV
import xgboost as xgb
# 定义超参数的网格
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1,0.2,0.3],
    'n_estimators': [50,100, 200],
    'subsample': [0.6,0.7, 0.8, 0.9,1],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2]
}

# 为每个目标量设置XGBRegressor
xgb_Regressors = [xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False) for _ in range(n_targets)]
# 为每个目标量运行网格搜索
for i in range(n_targets):
    # 包装XGBRegressor以使用GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_Regressors[i], param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error', verbose=1)
    # 训练模型并找到最佳参数
    grid_search.fit(x_train, y_train[:, i])
    # 打印最佳参数
    print(f'Target {i + 1} - Best parameters found: {grid_search.best_params_}')
    print(f'Target {i + 1} - Best score: {grid_search.best_score_}')
    print("---------------------finished----------------------")'''