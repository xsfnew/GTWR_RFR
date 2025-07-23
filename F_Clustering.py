import os
import numpy as np
from numpy import unique
from numpy import where
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt #matplotlib的一个子包
import seaborn as sns
import pandas as pd
import folium
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN  # 进行DBSCAN聚类
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # 计算 轮廓系数，CH 指标，DBI


# pip install tslearn
# tslearn和sklearn一样，是一款优秀的机器学习框架，tslearn更偏向于处理时间序列问题，
# 如其聚类模块就包含了DTW（Dynamic Time Warping）等算法及变种，也提供了轮廓系数对聚类效果评估，十分方便，使用文档。

# 轮廓系数提供了一个量化的聚类质量评估指标，它通过综合考虑簇内紧密度和簇间分离度，
# 给出了一个介于-1到1之间的值，值越大表示聚类效果越好。

# 1.亲和力传播聚类（层次聚类算法）
def model_AffinityPropagation(X):
    # 亲和力传播包括找到一组最能概括数据的范例。
    # 要调整的主要配置是将“阻尼”设置为0.5到1，甚至可能是“首选项”。
    # 其优点有：
    # 1.不需要制定最终聚类族的个数
    # 2.已有的数据点作为最终的聚类中心，而不是新生成一个族中心。
    # 3.模型对数据的初始值不敏感。
    # 4.对初始相似度矩阵数据的对称性没有要求。
    # 5.相比与k-centers聚类方法，其结果的平方差误差较小。
    # 在聚类的过程中有两种不同的信息进行交换，每一种信息代表一种竞争。
    # 需要注意的是，AP算法效率很低，测试数据不宜过大。

    print('亲和力传播聚类（AffinityPropagation）')
    # 定义模型
    model = AffinityPropagation(damping=0.9)
    # 匹配模型
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()


# 2.聚合聚类
def model_AgglomerativeClustering(X, num):
    # 聚合聚类涉及合并示例，直到达到所需的群集数量为止。
    # 它是层次聚类方法的更广泛类的一部分，主要配置是“n_clusters”集，这是对数据中的群集数量的估计。

    print('层次聚类方法（AgglomerativeClustering）')
    # 定义模型
    model = AgglomerativeClustering(n_clusters=num)
    # 模型拟合与聚类预测
    yhat = model.fit_predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()


# 3.BIRCH
def model_Birch(X, num):
    # BIRCH聚类（BIRCH是平衡迭代减少的缩写，聚类使用层次结构)包括构造一个树状结构，从中提取聚类质心。
    # BIRCH递增地和动态地群集传入的多维度量数据点，以尝试利用可用资源（即可用内存和时间约束）产生最佳质量的聚类。

    print('BIRCH聚类')
    # 定义模型
    model = Birch(threshold=0.01, n_clusters=num)
    # 适配模型
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()

def DBSCAN_Paramter(data):
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
    # plt.scatter(15, eps, color="r", s=5)
    # plt.plot([0, 15], [eps, eps], linestyle="--", color="r")
    # plt.plot([15, 15], [0, eps], linestyle="--", color="r")
    # plt.show()
    # print('eps: %d' % eps, '\tmin_samples: %d' % k)

    #######################################################################
    rs = []  # 存放各个参数的组合计算出来的模型评估得分和噪声比
    # eps_all = np.arange(14500, 16000, 500)  # eps参数从0.2开始到4，每隔0.2进行一次
    eps_all = np.arange(800, 2000, 100)  # eps参数从0.2开始到4，每隔0.2进行一次
    min_samples_all = np.arange(1, 3, 1)  # min_samples参数从2开始到20

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
                print('eps: %d' % eps, '\tmin_samples: %d' % min_samples, "\t轮廓系数:", format(score, '0.2'),
                      "\t噪声点占比:", format(raito, '.2%'), '\t分簇数:%d' % n_clusters_)
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
# 
# 4.DBSCAN聚类
def model_DBSCAN(X):
    # 是基于密度的空间聚类的噪声应用程序，涉及在域中寻找高密度区域，并将其周围的特征空间区域扩展为群集。
    # 依赖于基于密度的概念的集群设计，以发现任意形状的集群。并支持用户为其确定适当的值
    # 主要配置是“eps”距离和“min_samples”簇个数。
    print('DBSCAN聚类')
    # best_score_eps, best_score_min_samples = DBSCAN_Paramter(X)
    best_score_eps = 1300
    best_score_min_samples = 1
    
    # 定义模型
    model = DBSCAN(eps=best_score_eps, min_samples=best_score_min_samples, algorithm='kd_tree')
    # features = StandardScaler().fit_transform(X)  # 将特征进行归一化
    model.fit(X)
    labels = model.labels_  # 获取聚类之后没一个样本的类别标签

    # 模型拟合与聚类预测
    yhat = model.fit_predict(X)
    # 就是下面这个函数可以计算轮廓系数
    score = silhouette_score(X, labels)
    print(score)
      
    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print('eps: %d' % best_score_eps, '\tmin_samples: %d' % best_score_min_samples, "\t轮廓系数:", format(score, '0.2'),
        "\t噪声点占比:", format(raito, '.2%'), '\t分簇数:%d' % n_clusters_)
    
    
    # # 确定栅格大小
    # grid_size = 100
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # # 创建栅格
    # x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

    # # 计算每个栅格单元内的点数
    # density = np.zeros((grid_size, grid_size))
    # for point in X:
        # x_idx = int((point[0] - x_min) / (x_max - x_min) * grid_size)
        # y_idx = int((point[1] - y_min) / (y_max - y_min) * grid_size)
        # if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            # density[y_idx, x_idx] += 1

    # # 生成密度图
    # plt.figure(figsize=(8, 6))
    # plt.imshow(density, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
    # plt.colorbar(label='Point Density')
    # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', edgecolors='white', s=5, linewidths=2, label='Data Points')
    # plt.title('Grid Density Map of DBSCAN Clustering Results')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend(title='Cluster Labels')
    # plt.show()

    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        if cluster != -1:  # 排除噪声点
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1], s=2, label=f'Cluster {cluster}')
    
    # 绘制噪声点
    noise_ix = np.where(yhat == -1)
    plt.scatter(X[noise_ix, 0], X[noise_ix, 1], s=2, color='black', label='Noise')
    
    # 添加图例
    plt.legend()
    # 绘制散点图
    plt.show()

    return clusters


# 定义一个进行DBSCAN的函数
def DBSCAN_Cluster(X):
    """
    dbscan cluster
    :param X:  # 比如形状是（9434,4）表示9434个像素点
    :return:
    """
    db = DBSCAN(eps=0.01, min_samples=1000)
    try:
        features = StandardScaler().fit_transform(X)  # 将特征进行归一化
        db.fit(features)
    except Exception as err:
        print(err)
        ret = {
            'origin_features': None,
            'cluster_nums': 0,
            'db_labels': None,
            'cluster_center': None
        }
        return ret

    db_labels = db.labels_  # 获取聚类之后没一个样本的类别标签
    unique_labels = np.unique(db_labels)  # 获取唯一的类别

    num_clusters = len(unique_labels)
    cluster_centers = db.components_

    for cluster in unique_labels:
        # 获取此群集的示例的行索引
        row_ix = where(db_labels == unique_labels)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)        
    # 添加图例
    plt.legend()
    # 绘制散点图
    plt.show()

    ret = {
        'origin_features': features,  # (9434,4)
        'cluster_nums': num_clusters,  # 5  它是一个标量，表示5类，包含背景
        'db_labels': db_labels,  # (9434,)
        'unique_labels': unique_labels,  # (5,)
        'cluster_center': cluster_centers  # (6425,4)
    }

    return ret


# 画出聚类之后的结果
def plot_dbscan_result(features, db_labels, unique_labels, num_clusters):
    print('轮廓系数:' + silhouette_score(features, db_labels, metric='euclidean'))
    print('CH score:' + calinski_harabasz_score(features, db_labels))
    print('DBI:' + davies_bouldin_score(features, db_labels))

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, color in zip(unique_labels, colors):
        if k == -1:
            color = 'k'  # 黑色的，这代表噪声点

        index = np.where(db_labels == k)  # 获取每一个类别的索引位置
        x = features[index]

        plt.plot(x[:, 0], x[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % num_clusters)


# 5.k-means聚类
def model_KMeans(X, num):
    # K均值是聚类分析中广泛使用的方法。但是，仅当许多假设对数据集有效时，此方法才有效：
    # k-means 假设每个属性（变量）的分布方差是球形的;
    # 所有变量具有相同的方差;
    # 所有 k 个聚类的先验概率相同，即每个聚类的观测值数大致相等;
    # 如果违反了这3个假设中的任何一个，那么k均值将不正确。
    # 使用K均值时必须做出的重大决定是先验地选择聚类的数量。但是，正如我们将在下面看到的，此选择至关重要，并且对结果有很大的影响：
    # 是最常见的聚类算法，并涉及向群集分配示例，以尽量减少每个群集内的方差。
    # 描述一种基于样本将 N 维种群划分为 k 个集合的过程。
    # 主要配置是“n_clusters”超参数设置为数据中估计的群集数量。
    KMeans_SSE(X, num)
    KMeans_silhouette(X, num)

    print('k-means聚类')
    # 定义模型
    model = KMeans(n_clusters=num)
    # 模型拟合
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()
    return yhat


# 构造自定义函数，用于绘制不同k值和对应总的簇内离差平方和的折线图
def KMeans_SSE(X, clusters):
    # 选择连续的K种不同的值
    K = range(1, clusters + 1)
    # 构建空列表用于存储总的簇内离差平方和
    TSSE = []
    for k in K:
        # 用于存储各个簇内离差平方和
        SSE = []
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # 返回簇标签
        labels = kmeans.labels_
        # 返回簇中心
        centers = kmeans.cluster_centers_
        # 计算各簇样本的离差平方和，并保存到列表中
        for label in set(labels):
            SSE.append(np.sum((X.loc[labels == label,] - centers[label, :]) ** 2))
        # 计算总的簇内离差平方和
        TSSE.append(np.sum(SSE))

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制K的个数与GSSE的关系
    plt.plot(K, TSSE, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('簇内离差平方和之和')
    # 显示图形
    plt.show()


# 构造自定义函数，用于绘制不同k值和对应轮廓系数的折线图
def KMeans_silhouette(X, clusters):
    K = range(2, clusters + 1)
    # 构建空列表，用于存储个中簇数下的轮廓系数
    S = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        # 调用字模块metrics中的silhouette_score函数，计算轮廓系数
        S.append(silhouette_score(X, labels, metric='euclidean'))

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制K的个数与轮廓系数的关系
    plt.plot(K, S, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('轮廓系数')
    # 显示图形
    plt.show()


# 6.Mini-Batch K-均值
def model_MiniBatchKMeans(X, num):
    # 是K-均值的修改版本，它使用小批量的样本而不是整个数据集对群集质心进行更新，可以使大数据集的更新速度更快，并且可能对统计噪声更健壮。
    # 建议使用k-均值聚类的批量优化。与经典处理算法相比，这降低了计算成本的数量级，同时提供了比随机梯度下降更好的解决方案。
    # 主配置是“n_clusters”超参数，设置为数据中估计的群集数量。

    print('Mini-Batch K-均值')
    # 定义模型
    model = MiniBatchKMeans(n_clusters=num)
    # 模型拟合
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()


# 7.均值漂移聚类
def model_MeanShift(X):
    # 均值漂移聚类涉及到根据特征空间中的实例密度来寻找和调整质心。
    # 对离散数据证明了递推平均移位程序收敛到最接近驻点的基础密度函数，从而证明了它在检测密度模式中的应用。
    # 主要配置是“带宽”参数。

    print('均值漂移聚类（MeanShift）')
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=5000)
    print(bandwidth)
    # 定义模型
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # 模型拟合与聚类预测
    yhat = model.fit_predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()


# 8.OPTICS聚类
from sklearn.cluster import OPTICS


def model_OPTICS(X):
    # OPTICS 聚类（OPTICS 短于订购点数以标识聚类结构）是DBSCAN的修改版本。
    # 引入了一种新的算法，它不会显式地生成一个数据集的聚类，而是创建表示其基于密度的聚类结构的数据库的增强排序。
    # 此群集排序包含相当于密度聚类的信息，该信息对应于范围广泛的参数设置。
    # 主要配置是“eps”和“ min_samples”参数。

    print('OPTICS聚类')
    # 定义模型
    model = OPTICS(eps=0.8, min_samples=5)
    # 模型拟合与聚类预测
    yhat = model.fit_predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()


# 9.谱聚类spectral clustering
def model_SpectralClustering(X, num):
    # 谱聚类是一类通用的聚类方法，取自线性代数。
    # 这里，使用从点之间的距离导出的矩阵的顶部特征向量。
    # 要优化的是“n_clusters”参数，用于指定数据中的估计群集数量。

    print('谱聚类(SpectralClustering)')
    # 定义模型
    model = SpectralClustering(n_clusters=num)
    # 模型拟合与聚类预测
    yhat = model.fit_predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()


# 10.高斯混合模型
def model_GaussianMixture(X, num):
    # 高斯混合模型总结了一个多变量概率密度函数，就是混合了高斯概率分布。
    # 优化的主要配置是“n_clusters”参数，用于指定数据中估计的群集数量。

    print('高斯混合模型(GaussianMixture)')
    # 定义模型
    model = GaussianMixture(n_components=num)
    # 模型拟合
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=5)
    # 绘制散点图
    plt.show()


def display_cluster(clusters):
    ##############################################################
    df['cluster'] = clusters
    print('聚类结果图')
    plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='rainbow')
    # plt.scatter(df['longitude'], df['latitude'], c=clusters, cmap='rainbow')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Clustering of Coordinates')
    plt.show()

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
                            color='#C0C0C0' if clusters[i] == -1 else colors[clusters[i]], fill=True,
                            fill_color=colors[clusters[i]]).add_to(map_)

    map_.save('all_cluster.html')


if __name__ == "__main__":
    print('\r', end='')


    print('\r', end='')
    sns.set()

    data_dir = r'F:\Map\base'
    os.chdir(data_dir)

    print('导入数据...')
    df = pd.read_csv(data_dir + r'\modis_hn_Standardization.csv',
                     float_precision='round_trip')  # encoding方式可自行选择encoding='gbk',
    # df = df[['longitude', 'latitude','X', 'Y']].dropna(axis=0, how='all')
    # df = df[['X', 'Y']].dropna(axis=0, how='all')
    # data = np.array(df)
    ## 将经纬度转换为弧度，因为哈弗赛公式需要弧度作为输入
    # data = df[['longitude', 'latitude']].apply(lambda x: x.map(radians)).values
    # data=df[['X', 'Y']].dropna(axis=0, how='all')
    data = df[['X', 'Y']].values
    num = 5

    # clusters = model_AffinityPropagation(data)
    # clusters = model_AgglomerativeClustering(data,num)
    # clusters = model_Birch(data,num)

    clusters = model_DBSCAN(data)
    # # ret=DBSCAN_Cluster(data)  # 进行 DBSCAN聚类
    # # plot_dbscan_result(ret['origin_features'],ret['db_labels'],ret['unique_labels'],ret['cluster_nums']) # 展示聚类之后的结果

    # clusters = model_KMeans(data, num)
    # clusters = model_MiniBatchKMeans(data,num)
    # clusters = model_MeanShift(data)
    # clusters = model_OPTICS(data)
    # # clusters = model_SpectralClustering(data,num)
    # clusters = model_GaussianMixture(data,num)

    # display_cluster(clusters)
