<<<<<<< HEAD
=======
# 代码在微信公众号，不懂绘图中，请自己翻历史图文消息。
# 也接代做的，代做私信即可
# 相关运行问题请加qq群954990908
>>>>>>> 94aa6a0ee33af900a6c42c6c3ff753db01c77653
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from tqdm import tqdm
from joblib import Parallel, delayed
import shap
from sklearn.neighbors import KDTree
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Rbf
import joblib
from mgtwr.model import GWR, MGWR, GTWR, MGTWR, GTWRF


# 定义数据集的预处理
def load_data(file_path):
    """加载数据并提取坐标、特征和目标列"""
    data = pd.read_excel(file_path)
    coords = data[['latitude', 'longitude']]
    features = data.drop(columns=['ID', 'H', 'latitude', 'longitude'])
    target = data['H']
    nt = data['NT']
    return features, target, coords, nt


# 定义地理加权回归
# class GWR:
#     """地理加权回归模型（仅用于带宽选择）"""
#     def __init__(self, bandwidth=None):
#         self.bandwidth = bandwidth
#     def calculate_weights(self, coords, target_point):
#         """高斯核权重计算"""
#         distances = coords.apply(
#             lambda row: geodesic((row['latitude'], row['longitude']), target_point).meters,
#             axis=1
#         )
#         weights = np.exp(-0.5 * (distances / self.bandwidth)**2)
#         return weights.values  # 返回 NumPy 数组
#     def fit(self, X, y, coords):
#         """训练GWR模型（简化版，仅用于带宽选择）"""
#         # 将 DataFrame 转换为 NumPy 数组
#         X_np = X.values
#         y_np = y.values
#         coords_np = coords.values
#         self.coeffs = []
#         for idx in range(len(coords_np)):
#             target_point = (coords_np[idx][0], coords_np[idx][1])
#             weights = self.calculate_weights(coords, target_point)
#             # 使用 NumPy 数组进行矩阵运算
#             X_weighted = X_np * weights[:, np.newaxis]
#             y_weighted = y_np * weights
#             # 线性回归求解系数
#             coeff = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
#             self.coeffs.append(coeff)
#     def predict(self, X, coords):
#         """预测（简化示例）"""
#         X_np = X.values  # 转换为 NumPy 数组
#         return np.array([np.dot(X_np[i], self.coeffs[i]) for i in range(len(coords))])

# 定义最优宽带计算方式
def optimize_bandwidth_gwr(X_train, y_train, coords_train, bandwidth_candidates):
    """基于GWR的交叉验证选择最优带宽"""
    best_bw = None
    best_rmse = float('inf')
    for bw in bandwidth_candidates:
        model = GWR(bandwidth=bw)
        kf = KFold(n_splits=3)
        rmse_scores = []
        for train_idx, val_idx in kf.split(X_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], coords_train.iloc[train_idx])
            y_pred = model.predict(X_train.iloc[val_idx], coords_train.iloc[val_idx])
            rmse = np.sqrt(mean_squared_error(y_train.iloc[val_idx], y_pred))
            rmse_scores.append(rmse)
        avg_rmse = np.mean(rmse_scores)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_bw = bw
    return best_bw


# 定义随机森林参数
def optimize_rf_params(X_train, y_train):
    """网格搜索优化随机森林参数"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [3, 5, 7]
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=12),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


# 定义空间权重
def compute_spatial_weights(coords, bandwidth):
    """基于GWR模型计算所有点的空间权重矩阵"""
    gwr = GWR(bandwidth=bandwidth)
    weights_matrix = []
    for idx in range(len(coords)):
        target_point = (coords.iloc[idx]['latitude'], coords.iloc[idx]['longitude'])
        weights = gwr.calculate_weights(coords, target_point)
        weights_matrix.append(weights)
    return np.array(weights_matrix)


def _fit_single_optimized(idx, X_np, y_np, weights, rf_params):
    """优化后的单模型训练函数"""
    model = RandomForestRegressor(**rf_params)
    model.fit(X_np, y_np, sample_weight=weights)
    return idx, model


def build_gwrf_model(t, X_train, y_train, coords_train, best_bandwidth, tau, best_rf_params):
    """初始化并训练GWRF模型"""
    gwrf = GTWRF(t, coords_train, X_train, y_train, bandwidth=best_bandwidth, tau=tau, rf_params=best_rf_params)
    # gwrf.fit(X_train, y_train, coords_train, n_jobs=4)
    gwrf.fit(n_jobs=4, _fit_single_optimized=_fit_single_optimized)
    return gwrf


def validate_model(model, X_val, y_val, coords_val):
    """在验证集上评估模型性能"""
    y_pred = model.predict(X_val, coords_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    return {'RMSE': rmse, 'R2': r2}


# 导入数据
nt, features, target, coords = load_data('F:/20216.xlsx')
nt1, features1, target1, coords1 = load_data('F:/20218.xlsx')
# 划分数据集
# 训练集
(nt_train, X_train, y_train, coords_train) = (nt, features, target, coords)
# 测试集
(nt_val, X_val, y_val, coords_val) = (nt1, features1, target1, coords1)

# 优化随机森林参数
best_rf_params = {'max_depth': 14, 'min_samples_split': 2, 'n_estimators': 300}  # optimize_rf_params(X_train, y_train)
# # 使用GTWR找最佳带宽
# bandwidth_candidates = list(range(3850, 3951))  # 3850到3950，步长为1
# best_bandwidth = optimize_bandwidth_gwr(X_train, y_train, coords_train, bandwidth_candidates)
bw = 7639.3  # 7721.4  # 8307  # 12569.5
tau = 5

# 训练GWRF模型
gwrf_model = build_gwrf_model(nt_train, coords_train, X_train, y_train, bw, best_rf_params)
# 对测试集进行预测
y_test_pred = gwrf_model.predict(X_val, coords_val)

# 检查预测值中的NA值这部分注意视频讲解
# print(f"预测值中包含NA的数量: {np.isnan(y_test_pred).sum()}")
# 创建包含预测值和原始数据的DataFrame
# results_df = pd.DataFrame({
#    'True_H': y_val.values,
#    'Predicted_H': y_test_pred
# })
# 添加坐标信息
# full_results_df = pd.concat([
#    coords_val.reset_index(drop=True),
#    X_val.reset_index(drop=True),
#    results_df
# ], axis=1)
# 删除包含NA值的行
# clean_results_df = full_results_df.dropna(subset=['Predicted_H'])
# 分离出清理后的数据
# y_test_pred = clean_results_df['Predicted_H'].values
# y_val = clean_results_df['True_H'].values
# coords_val = clean_results_df[['latitude', 'longitude']]
# X_val = clean_results_df.drop(columns=['True_H', 'Predicted_H', 'latitude', 'longitude'])
# 计算清理后的性能指标
# if len(y_val) > 0:
#    test_rmse = np.sqrt(mean_squared_error(y_val, y_test_pred))
#    test_r2 = r2_score(y_val, y_test_pred)
#    print(f"清理后的测试集大小: {len(y_val)}/{len(y_val)} 样本")
#    print(f"清理后的测试指标: RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
# else:
#    print("警告: 清理后没有剩余样本!")

# 计算性能指标
test_rmse = mean_squared_error(y_val, y_test_pred)
test_r2 = r2_score(y_val, y_test_pred)
print(f"Test Metrics: RMSE={test_rmse:.2f}, R²={test_r2:.2f}")
# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_test_pred, alpha=0.6)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)],
         '--', color='red', lw=2)  # 添加对角线
plt.xlabel('True H Values')
plt.ylabel('Predicted H Values')
plt.grid(True)
plt.show()
