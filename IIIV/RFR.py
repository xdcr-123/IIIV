import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import palettable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
from sklearn.inspection import PartialDependenceDisplay
import shap
from scipy.interpolate import splev, splrep
from sklearn.inspection import partial_dependence

# 读取数据
data1 = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\Databse_IIIV.xlsx")
data1 = data1.drop(data1.index[0])    #删除第一行数据，并非是列名，这里删除第一行数据降低损失
data2 = data1.values                 #将DataFrame转成pandas数据结构
feature_names = data1.columns[:-2].tolist()
X = data2[:, :-2]
Y = data2[:, -2:]
scaler = MinMaxScaler()#归一化
X = scaler.fit_transform(X)
# 将归一化后的数据转换为DataFrame
X_df = pd.DataFrame(X, columns=feature_names)
#X_df.to_excel("E:\excel-data\\alkali_metal_adsorption_normalization.xlsx", index=False)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=70)
# 将归一化后的数据转换为DataFrame以便查看
x_train_df = pd.DataFrame(x_train, columns=feature_names)
x_test_df = pd.DataFrame(x_test, columns=feature_names)
print("归一化后的训练集前五行:")
print(x_train_df.head())
print("\n归一化后的测试集前五行:")
print(x_test_df.head())



# 定义两组不同的超参数
param_grid = [
    {
        'max_depth': 10,
        'n_estimators': 30,
        'bootstrap': True
    },
    {
        'max_depth': 10,
        'n_estimators': 60,
        'bootstrap': True
    }
]

# 训练两个模型，每个模型使用不同的超参数
models = []
y_train_preds = []
y_test_preds = []
losses = {
    'r2': [],
    'mean_squared_error': [],
    'mean_absolute_error': []
         }
for i in range(y_train.shape[1]):
    params = param_grid[i]
    rf = RandomForestRegressor(**params, random_state= 42)
    rf.fit(x_train, y_train[:, i])
    models.append(rf)
    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
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

'''
#部分依赖图(带ICE)
#确保执行pdp之前特征名字符串之间不存在空格,空格用下划线代替
sns.set_style("darkgrid")
selected_features_model1 = ['Eea-3', 'D-1', 'N-1','Eea-1','X-3','Y1']
selected_features_model2 = ['N-1', 'D-1', 'X-3', 'Eea-3', 'N-3', 'D-3']
# 绘制特定特征的部分依赖图的函数
def plot_selected_features_partial_dependence(model, X_df, selected_features, layout):
    # 创建图形对象并设置大小
    figsize_width = 3 * layout[1]
    figsize_height = 3 * layout[0]
    fig, axs = plt.subplots(layout[0], layout[1], figsize=(figsize_width, figsize_height))
    axs = axs.flatten()  # 将axs数组展平，方便迭代

    for ax, feature in zip(axs, selected_features):
        ax.set_aspect('equal')  # 确保子图是正方形
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_df,
            features=[feature],
            ax=ax,
            kind='average',
            centered=True,
            subsample=100,
            grid_resolution=50,
            pd_line_kw={
                'color': '#ee1d21',  # 设置 PD 线的颜色
                'linewidth': 2,  # 设置 PD 线的粗细
            },
        ice_lines_kw = {
            'color': 'black',  # 设置 ice线的颜色
            'linewidth': 1,  # 设置 ice 线的粗细
        }
        )

        ax.set_xlabel(feature, fontsize=28)  # 设置 x 轴标签
        ax.set_ylabel('Partial Dependence', fontsize=28)  # 设置 y 轴标签
        ax.tick_params(axis='both', labelsize=28)  # 设置刻度字体大小
    # 调整子图之间的间距
    fig.tight_layout(w_pad=0.6, h_pad=2)
    plt.show()

# 调用函数绘制模型的特定特征的部分依赖图
plot_selected_features_partial_dependence(models[0], X_df, selected_features_model1,  layout=(2, 3))
plot_selected_features_partial_dependence(models[1], X_df, selected_features_model2,  layout=(2, 3))
'''

#全部特征pdp
'''fig, ax = plt.subplots(figsize=(10, 6))   
display = PartialDependenceDisplay.from_estimator(
    models[0],  # 假设我们绘制第一个模型的部分依赖图
    x_train_df,  # 使用训练数据
    features=feature_names,
    ax=ax,
    grid_resolution=20  # 分辨率，可以根据需要调整
)
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=3)
plt.show()'''''

#特征值对shap值的影响（交互shap特征排序图）
#纵轴按照所有样本的SHAP值之和对特征排序，横轴是SHAP值（特征对模型输出的影响分布）；
#每个点代表一个样本，样本量在纵向堆积，颜色表示特征值（红色对应高值，蓝色对应低值）
explainer = shap.TreeExplainer(models[0])
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train, feature_names=feature_names,plot_size=0.3)
explainer = shap.TreeExplainer(models[1])
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train, feature_names=feature_names,plot_size=0.3)



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
    'size': 40,
    'weight': 'normal',  # 或者使用 'heavy', 'light', 'normal', 'ultrabold' 等
}
sns.jointplot(data=data1, x='Actual', y='Predicted', kind='scatter', hue='Dataset',
              palette={"Train": colors[0], "Test": colors[1]}, height=6, ratio=10)
min_val = min(np.min(data1['Actual']), np.min(data1['Predicted']))
max_val = max(np.max(data1['Actual']), np.max(data1['Predicted']))
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=4, label='Perfect Fit')
plt.xlabel('DFT-calculated Eb', fontdict)
plt.ylabel('Predicted Eb', fontdict)
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
#plt.title('Actual vs Predicted Target 1 Values', fontsize=16)
plt.legend(loc='upper left',prop={'size': 25, 'family': 'Times New Roman'})
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

#不带边际分布的散点图
'''sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
colors = ['#2da7e2', '#a5aaa3']  # 颜色列表，用于区分不同的目标
for i, model in enumerate(models):
    plt.subplot(1, 2, i+1)
    plt.scatter(y_train[:, i], models[i].predict(x_train), alpha=0.7, c=colors[i], edgecolor='none', s=75, label='Train')
    plt.scatter(y_test[:, i], models[i].predict(x_test), alpha=0.7, c='#ff807f', edgecolor='none', s=75, label='Test')
    plt.plot([min(y_test[:, i]), max(y_test[:, i])],
             [min(y_test[:, i]), max(y_test[:, i])],
             color='yellow', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.xlabel('Actual Target Value', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Target Value', fontsize=14, fontweight='bold')
    plt.title(f'Actual vs Predicted Target {i+1} Values', fontsize=16)
    plt.legend()
plt.tight_layout()
plt.show()'''

# 绘制回归预测曲线
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

# 绘制特征重要性图
plt.figure(figsize=(12, 6))
for i, model in enumerate(models):
    target_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': target_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    ax = plt.subplot(1, 2, i + 1)
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
    ax.set_title(f'Feature Importance for Target {i + 1}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
plt.tight_layout()
plt.show()


'''
new_data = pd.read_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict.xlsx")
x_val = new_data.iloc[:,:].values
dtest = rf.DMatrix(x_val)
y_pred_ads = models[0].predict(dtest)
y_pred_wf = models[1].predict(dtest)
new_data['Predicted_adsorption'] = y_pred_ads
new_data['Predicted_work_function'] = y_pred_wf
new_data.to_excel("C:\\Users\Dell\Desktop\IIIV\\Verification_Predict_result.xlsx", index=False)
'''
