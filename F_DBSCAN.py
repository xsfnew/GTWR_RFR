import os
import pandas as pd
import numpy as np
from math import radians
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import folium

print('\r', end='')
sns.set()

data_dir = r'F:\Map\base'
os.chdir(data_dir)

print('导入数据...')
df = pd.read_csv(data_dir + r'\modis_hn_Standardization.csv', float_precision='round_trip')
# df = df[['longitude', 'latitude','X', 'Y']].dropna(axis=0, how='all')
# df = df[['X', 'Y']].dropna(axis=0, how='all')
# data = np.array(df)
## 将经纬度转换为弧度，因为哈弗赛公式需要弧度作为输入
# data = df[['longitude', 'latitude']].apply(lambda x: x.map(radians)).values
# data=df[['X', 'Y']].dropna(axis=0, how='all')
data = df[['X', 'Y']].values


def FindParamter(data):
    print('选择参数')
    # def select_MinPts(data, k):
    #     k_dist = []
    #     for i in range(data.shape[0]):
    #         dist = (((data[i] - data) ** 2).sum(axis=1) ** 0.5)
    #         dist.sort()
    #         k_dist.append(dist[k])
    #     return np.array(k_dist)
    #
    #
    # k = 2  # 此处k取 2*2 -1
    # k_dist = select_MinPts(data, k)
    # k_dist.sort()
    # plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
    #
    # # 由拐点确定邻域半径
    # eps = k_dist[::-1][15]
    # plt.scatter(15, eps, color="r")
    # plt.plot([0, 15], [eps, eps], linestyle="--", color="r")
    # plt.plot([15, 15], [0, eps], linestyle="--", color="r")
    # plt.show()
    # print('eps: %d' % eps, '\tmin_samples: %d' % k)

    #######################################################################
    rs = []  # 存放各个参数的组合计算出来的模型评估得分和噪声比
    # eps_all = np.arange(1000, 3000, 100)  # eps参数从0.2开始到4，每隔0.2进行一次
    # min_samples_all = np.arange(5, 15, 1)  # min_samples参数从2开始到20
    eps_all = np.arange(20000, 30000, 1000)  # eps参数从0.2开始到4，每隔0.2进行一次
    min_samples_all = np.arange(20, 30, 1)  # min_samples参数从2开始到20

    best_score = 0
    best_score_eps = 0
    best_score_min_samples = 0

    for eps in eps_all:
        for min_samples in min_samples_all:
            try:  # 因为不同的参数组合，有可能导致计算得分出错，所以用try
                db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree').fit(data)
                labels = db.labels_  # 得到DBSCAN预测的分类便签
                score = metrics.silhouette_score(data, labels)  # 轮廓系数评价聚类的好坏，值越大越好
                raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
                rs.append([eps, min_samples, score, raito, n_clusters_])
                print('eps: %d' % eps, '\tmin_samples: %d' % min_samples, "\t轮廓系数:", format(score, '0.2%'),
                      "\t噪声点个数占比:", format(raito, '.2%'), '\t分簇的数目:%d' % n_clusters_)
                if score > best_score:
                    best_score = score
                    best_score_eps = eps
                    best_score_min_samples = min_samples
            except:
                db = ''  # 这里用try就是遍历eps，min_samples 计算轮廓系数会出错的，出错的就跳过
            else:
                db = ''
    rs = pd.DataFrame(rs)
    rs.columns = ['eps_all', 'min_samples_all', 'score', 'raito', 'n_clusters']

    plt.plot(rs['eps_all'], rs['score'])
    plt.xlabel('Number of eps')
    plt.ylabel('Sihouette Score')
    plt.plot(rs['min_samples_all'], rs['score'])
    plt.xlabel('Number of min_samples')
    plt.ylabel('Sihouette Score')
    plt.show()
    print(rs[rs['score'] == rs['score'].max()])

    # 聚类指标
    # from sklearn.metrics.cluster import rand_score 
    # from sklearn.metrics.cluster import adjusted_rand_score 
    # rand_score(data, labels)# RI 兰德指数
    # adjusted_rand_score(data, labels)# ARI 调整兰德指数
    # #NMI
    # from sklearn.metrics.cluster import normalized_mutual_info_score
    # normalized_mutual_info_score(
    # labels_true,
    # labels_pred,
    # *,
    # average_method='arithmetic',
    # )
    # #Jaccard系数
    # from sklearn.metrics import jaccard_score
    # jaccard_score(
    # y_true,
    # y_pred,
    # *,
    # labels=None,
    # pos_label=1,
    # average='binary',
    # sample_weight=None,
    # zero_division='warn',
    # )
    # #轮廓系数
    # from sklearn.metrics.cluster import silhouette_score
    # silhouette_score(    data, labels,    *,    metric='euclidean',
    # # 在数据的随机子集上计算轮廓系数时要使用的样本大小
    # sample_size=None,
    # random_state=None,
    # **kwds,
    # )
    # CH指标
    # from sklearn.metrics.cluster import calinski_harabasz_score
    # calinski_harabasz_score(data, labels)
    return best_score_eps, best_score_min_samples


############################################################
# best_score_eps, best_score_min_samples = FindParamter(data)
best_score_eps = 29000
best_score_min_samples = 25
# eps: 10500      min_samples: 29         平均轮廓系数: -1.25%    噪声点个数占总数的比例: 6.71%   分簇的数目:16
# eps: 13000      min_samples: 20         平均轮廓系数: -0.66%    噪声点个数占总数的比例: 1.30%   分簇的数目:8
# eps: 13500      min_samples: 20         平均轮廓系数: 3.93%     噪声点个数占总数的比例: 0.98%   分簇的数目:8
# eps: 14000      min_samples: 23         平均轮廓系数: 9.77%     噪声点个数占总数的比例: 1.15%   分簇的数目:8
print('使用DBSCAN算法')  # leaf_size=30, algorithm='kd_tree',"ball_tree", "kd_tree", "brute"
dbscan = DBSCAN(eps=best_score_eps, min_samples=best_score_min_samples, algorithm='kd_tree')
clusters = dbscan.fit_predict(data)

raito = len(clusters[clusters[:] == -1]) / len(clusters)  # 计算噪声点个数占总数的比例
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)  # 获取分簇的数目
# 模型评估   评估标准轮廓系数法（Silhouette Cofficient），用来评估聚类算法的效果。
score = metrics.silhouette_score(data, clusters)  # 轮廓系数
print('eps: %d' % best_score_eps, '\tmin_samples: %d' % best_score_min_samples, "\t轮廓系数:",
      format(score, '0.2'),
      "\t噪声点个数占比:", format(raito, '.2%'), '\t分簇的数目:%d' % n_clusters_)
##############################################################
df['cluster'] = clusters
print('聚类结果图')
plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='rainbow')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('DBSCAN Clustering of Coordinates')
plt.show()

################# 聚类结果可视化 #########################
# 从SciPy中导入dendrogram函数和ward聚类函数
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)

# 将ward聚类应用于数据数组X
# SciPy的ward函数返回一个数组，指定执行凝聚聚类时跨越的距离
linkage_array = ward(X)

# 现在为包含簇之间距离的linkage_array绘制树状图
dendrogram(linkage_array)

# 在树中标记划分成两个簇或三个簇的位置
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")

###############################################################
print('地图可视化，输出HTML')
sns.lmplot(x='longitude', y='latitude', data=df, hue='cluster', fit_reg=False)

map_ = folium.Map(location=[25.974729, 112.301663], zoom_start=7,
                  # tiles='http://webst04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                  tiles='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                  # tiles='https://t0.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=c269037ee6a5dec7ffa9942e8ad9e3a2',
                  attr='default')

colors = ['#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#DC143C',
          '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#DC143C',
          '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#000000']

data1 = df[['longitude', 'latitude']].values
for i in range(len(data1)):
    folium.CircleMarker(location=[data1[i][1], data1[i][0]],
                        radius=1, popup=clusters[i],
                        color='#C0C0C0' if clusters[i] == -1 else colors[clusters[i % len(colors)]], fill=True,
                        fill_color=colors[clusters[i % len(colors)]]).add_to(map_)

map_.save('all_cluster_DBSCAN.html')
