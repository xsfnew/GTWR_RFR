import os
import datetime as dt
import numpy as np
from numpy import polyfit, poly1d
import pandas as pd
import geopandas as gpd
# import contextily as ctx
from geopandas import GeoDataFrame
from shapely.geometry import Point
from pylab import figure, scatter, show
import esda
from esda.moran import Moran_Local, Moran
from esda.getisord import G_Local
from splot.esda import plot_moran
from splot.esda import lisa_cluster
from splot.esda import moran_scatterplot
from splot.esda import plot_local_Probably
from splot.esda import plot_local_autocorrelation
from splot._viz_libpysal_mpl import plot_spatial_weights
import libpysal
import libpysal as lps
from libpysal.weights.contiguity import Queen
from libpysal import examples
from giddy.directional import Rose
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import ticker
from matplotlib import colors
import matplotlib.font_manager as fm

# pip install pysal

# geopandas读取CSV文件
def csv_to_point_shp(csv_file):
    df = pd.read_csv(csv_file, float_precision='round_trip')  # encoding方式可自行选择encoding='gbk',
    gdf = gpd.GeoDataFrame(df, crs='EPSG:3857',
                           geometry=gpd.points_from_xy(df.X, df.Y))  # lng,lat根据实际字段选择
    # gdf = gpd.GeoDataFrame(df, crs='EPSG:4326',
    #                       geometry=gpd.points_from_xy(df.longitude, df.latitude))  # lng,lat根据实际字段选择
    return gdf, df


# 日期转换为1-365天数序号
def out_day_by_date(date):
    '''
    根据输入的日期计算该日期是在当年的第几天
    '''
    year = date.year
    month = date.month
    day = date.day
    months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if 0 < month <= 12:
        sum = months[month - 1]
    else:
        print("month error")
    sum += day
    leap = 0
    # 接下来判断平年闰年
    if (year % 400 == 0) or ((year % 4) == 0) and (year % 100 != 0):  # and的优先级大于or
        # 1、世纪闰年:能被400整除的为世纪闰年
        # 2、普通闰年:能被4整除但不能被100整除的年份为普通闰年
        leap = 1
    if (leap == 1) and (month > 2):
        sum += 1  # 判断输入年的如果是闰年,且输入的月大于2月,则该年总天数加1
    return sum


def model_gaussian_kde1(x, y):
    print("核密度估计")
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # 对数据进行排序
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # 绘制密度散点图
    scatter = ax.scatter(x, y, c=z, cmap="bwr", s=5)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density')

    # 设置标题和标签
    plt.title("Density Scatter Plot")
    plt.xlabel("X Label")
    plt.ylabel("Y Label")

    # 显示图形
    plt.show()

# 核密度分析（Kernel Density Estimation, KDE）
# 计算二维点（x，y）坐标的密度，并绘制散点图。
def model_gaussian_kde(x, y, contour=False):
    """
    参数:
    x -- x坐标的数组
    y -- y坐标的数组
    """
    print("核密度估计")
    # 将x和y坐标组合成一个二维数组
    points = np.vstack([x, y])

    # 使用gaussian_kde计算密度估计
    kde = gaussian_kde(points,bw_method='silverman')


    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    scatter = ax.scatter(x, y, c=kde(points), cmap="bwr", s=5)

    # 添加颜色条
    plt.colorbar(scatter, label='Density')
    # # 添加颜色条
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Density')
    
    if contour==True:
        # 创建一个网格来评估密度
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        # 使用kde函数计算网格点的密度
        Z = kde(positions).reshape(X.shape)
        # 绘制密度等高线
        plt.contour(X, Y, Z, levels=10, colors='black', linestyles='dashed', label='Density Contours')

    # 设置图表标题和标签
    plt.title('Gaussian KDE Density Estimation')
    plt.xlabel('lng')
    plt.ylabel('lat')
    plt.legend()

    # 显示图表
    plt.show()


#3. 信息熵分析
def model_entropy(x, y):
    print("信息熵分析")
    # 使用 zip 函数将 x 和 y 组合成一个元组的列表
    coords_tuple_list = list(zip(x, y))
    # 将元组的列表转换为 numpy 二维数组
    coords = np.array(coords_tuple_list)

    # 计算每个点的邻域内点的数量
    def count_neighbors(coords, radius):
        n = len(coords)
        counts = np.zeros(n)
        for i in range(n):
            distances = np.linalg.norm(coords - coords[i], axis=1)
            counts[i] = np.sum(distances <= radius)
        return counts

    # 计算信息熵
    def calculate_entropy(counts):
        p = counts / np.sum(counts)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    # 计算每个点的邻域内点的数量
    radius = 5000  # 1 km

    counts = count_neighbors(coords, radius)
    # 计算信息熵
    entropy = calculate_entropy(counts)
    print("信息熵：", entropy)


#统计分析-单位面积内火点个数比来计算密度
def model_counts_areas(x, y):
    # 加载火点坐标数据
    gdf = gpd.GeoDataFrame.from_file('fire_points.shp')  # 假设火点数据存储在 shapefile 文件中

    # 加载研究区域的行政区划数据
    regions = gpd.GeoDataFrame.from_file('regions.shp')  # 假设行政区划数据存储在 shapefile 文件中

    # 计算每个区域内的火点数量
    fire_counts = gpd.sjoin(gdf, regions, op='within').groupby('region_id').size()

    # 计算每个区域的面积
    region_areas = regions['geometry'].area

    # 计算火点的发生密度
    fire_density = fire_counts / region_areas

    # 可视化结果
    fig, ax = plt.subplots(1, figsize=(12, 10))
    regions.plot(column=fire_density, cmap='Blues', edgecolor='white', scheme='fisher_jenks', k=5, ax=ax, legend=True)
    ax.set_title('林火发生密度分布')
    plt.show()


def plot_local_Probably(moran_loc, gdf, fieldName, p=0.05):
    # 自定义绘制Local Moran's I显著性分布图
    fig, ax = plt.subplots(figsize=(10, 10))
    significant = moran_loc.p_sim < p
    gdf[significant].plot(column=moran_loc.p_sim[significant], cmap='Blues', edgecolor='white', ax=ax, legend=True)
    ax.set_title(f'{fieldName} Local Moran\'s I Significant Distribution (p<{p})')
    return fig, ax


def fire_Moran():
    import warnings

    warnings.filterwarnings("ignore")

    ##设置控制台显示格式
    np.set_printoptions(threshold=5)  # 显示所有行数据np.inf
    pd.set_option('display.max_columns', None)  # 显示所有列

    fieldName = "probability" #area_Proba,Fire_Count,probability
    root_dir = r'F:\Map\base'
    os.chdir(root_dir)

    print('读取数据...')
    gdf, df = csv_to_point_shp('modis_hn_Standardization.csv') 
    # 核密度估计
    # model_gaussian_kde(gdf['X'], gdf['Y'])
    model_entropy(gdf['X'], gdf['Y'])
    exit()

    fieldValue = gdf[fieldName]
    # fieldValue = gdf[fieldName].values
    ## fieldValue = [(out_day_by_date(dt.datetime.strptime(dn, '%Y-%m-%d')) - 1) for dn in df['road_dist'].values]
    ## fieldValue = gdf['road_dist'].values
    # gdf[fieldName] = fieldValue

    print("创建权重矩阵...")
    # 从地理数据框架 df 中创建Queen 邻接权重矩阵（spatial w matrix）。
    # w = Queen.from_dataframe(gdf)
    # w = libpysal.weights.Rook.from_dataframe(gdf)  # 使用Rook式邻接矩阵 # 邻接权重
    # w = libpysal.weights.Kernel.from_dataframe(df=gdf, ids='OBJECTID')
    # w = libpysal.weights.distance.Kernel.from_dataframe(gdf, fixed=False, k=10) #核密度权重
    # w = libpysal.weights.KNN.from_dataframe(df=gdf, ids='OBJECTID') # k最近邻权重
    w = libpysal.weights.DistanceBand.from_dataframe(gdf, threshold=10000, binary=False)  # 距离阈值权重
    print("权重矩阵weights转换方式为行标准化（row-standardized）...")
    w.transform = 'r'

    # 计算 Getis-Ord-Gi* 统计量
    g_local = G_Local(fieldValue, w)
    fig, ax = plt.subplots(1, figsize=(12, 10))
    gdf.plot(column=g_local.p_sim, cmap='Blues', edgecolor='white',  scheme='fisher_jenks', k=5, ax=ax, legend=True)
    ax.set_title('林火热点和冷点分布')
    plt.show()
    
    # plot_spatial_weights(w, gdf)
    # plt.show()

    print("Moran's I指数计算...")
    moran = Moran(fieldValue, w)
    print("Moran's I 值为：", moran.I)
    print("随机分布假设下Z检验值为：", moran.z_rand)
    print("随机分布假设下Z检验的P值为：", moran.p_rand)
    print("正态分布假设下Z检验值为：", moran.z_norm)
    print("正态分布假设下Z检验的P值为：", moran.p_norm)

    print("绘制Moral's I散点图")
    fig, ax = moran_scatterplot(moran, p=0.05, zstandard=True)  # ,aspect_equal=True)
    ax.set_xlabel('Wildfire probability')
    ax.set_ylabel('Spatial Lag of Wildfire probability')
    plt.show()

    print("绘制Moran's I指数图")
    fig, ax = plot_moran(moran, zstandard=True, figsize=(11, 6))
    plt.show()

    print("Local Moran's I指数计算...")
    moran_loc = Moran_Local(fieldValue, w)

    ##xy = [Point(xy) for xy in zip(gpd.GeoSeries.x, gpd.GeoSeries.y)]
    ##pts = gpd.GeoSeries(xy)  # 创建点要素数据集
    # data1 = pd.DataFrame(moran_loc)
    # data1.to_csv('a.csv')
    ##data1.to_file('a.shp', driver='ESRI Shapefile', encoding='utf-8')

    print("绘制聚集区空间分布图")
    fig, ax = lisa_cluster(moran_loc, gdf, p=0.05, figsize=(10, 10), marker='.', markersize=5)
    plt.show()

    # print("绘制结果组合图")
    # fig, ax = plot_local_autocorrelation(moran_loc, gdf, fieldName, p=0.05)
    # plt.show()

    print("绘制显著性分布图")
    fig, ax = plot_local_Probably(moran_loc, gdf, fieldName, p=0.05)
    plt.show()
 
def GTWRF_Result_Moran():
    import warnings

    warnings.filterwarnings("ignore")

    np.set_printoptions(threshold=5)
    pd.set_option('display.max_columns', None)

    shp_path = r'F:\Map\base\GTWRF_Result.shp'
    print('读取数据...')
    gdf = gpd.read_file(shp_path)

    # 如果需要坐标转换，取消注释以下代码
    # gdf = gdf.to_crs('EPSG:3857')

    # 提取坐标信息
    gdf['X'] = gdf.geometry.x
    gdf['Y'] = gdf.geometry.y

    # 核密度估计或熵模型（根据需要取消注释）
    # model_gaussian_kde(gdf['X'], gdf['Y'])
    # model_entropy(gdf['X'], gdf['Y'])

    fieldValue = gdf['probabilit'] - gdf['predict_va']

    print("创建权重矩阵...")
    # 根据需要选择合适的权重矩阵
    w = libpysal.weights.DistanceBand.from_dataframe(gdf, threshold=10000, binary=False)
    w.transform = 'r'

    # print("计算 Getis-Ord-Gi* 统计量...")
    # g_local = G_Local(fieldValue, w)
    # fig, ax = plt.subplots(1, figsize=(12, 10))
    # gdf.plot(column=g_local.p_sim, cmap='Blues', edgecolor='white', scheme='fisher_jenks', k=5, ax=ax, legend=True)
    # ax.set_title('热点和冷点分布')
    # plt.show()

    print("计算 Moran's I指数...")
    moran = Moran(fieldValue, w)
    print(f"Moran's I 值为：{moran.I}")
    print(f"随机分布假设下Z检验值为：{moran.z_rand}")
    print(f"随机分布假设下Z检验的P值为：{moran.p_rand}")
    print(f"正态分布假设下Z检验值为：{moran.z_norm}")
    print(f"正态分布假设下Z检验的P值为：{moran.p_norm}")

    print("绘制 Moran's I散点图...")
    fig, ax = moran_scatterplot(moran, p=0.05, zstandard=True)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Spatial Lag of Variable')
    plt.show()

    # print("绘制 Moran's I指数图...")
    # fig, ax = plot_moran(moran, zstandard=True, figsize=(11, 6))
    # plt.show()

    # print("计算 Local Moran's I指数...")
    # moran_loc = Moran_Local(fieldValue, w)

    # print("绘制聚集区空间分布图...")
    # fig, ax = lisa_cluster(moran_loc, gdf, p=0.05, figsize=(10, 10), marker='.', markersize=5)
    # plt.show()

    # print("绘制显著性分布图...")
    # fig, ax = plot_local_Probably(moran_loc, gdf, fieldName, p=0.05)
    # plt.show()

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    np.set_printoptions(threshold=5)
    pd.set_option('display.max_columns', None)

    shp_path = r'F:\Map\base\GTWRF_Result.shp'
    print('读取数据...')
    gdf = gpd.read_file(shp_path)

    # 如果需要坐标转换，取消注释以下代码
    # gdf = gdf.to_crs('EPSG:3857')

    # 提取坐标信息
    gdf['X'] = gdf.geometry.x
    gdf['Y'] = gdf.geometry.y

    # 核密度估计或熵模型（根据需要取消注释）
    # model_gaussian_kde(gdf['X'], gdf['Y'])
    # model_entropy(gdf['X'], gdf['Y'])
    fields=['cof_N_V_tm','cof_N_V_pr','cof_N_V_wi','cof_N_V_SL','cof_N_V_DE',
        'cof_N_V_AS','cof_YU_BI_','cof_PINGJU','cof_N_V_GD','cof_N_V_PO',
        'cof_N_road','cof_N_wate']

    print("创建权重矩阵...")
    # 根据需要选择合适的权重矩阵
    w = libpysal.weights.DistanceBand.from_dataframe(gdf, threshold=10000, binary=False)
    w.transform = 'r'

    for field in fields:
        # print("计算 Getis-Ord-Gi* 统计量...")
        # g_local = G_Local(fieldValue, w)
        # fig, ax = plt.subplots(1, figsize=(12, 10))
        # gdf.plot(column=g_local.p_sim, cmap='Blues', edgecolor='white', scheme='fisher_jenks', k=5, ax=ax, legend=True)
        # ax.set_title('热点和冷点分布')
        # plt.show()

        fieldValue = gdf[field]
        print(f"计算{field} —— Moran's I指数...")
        moran = Moran(fieldValue, w)
        print(f"Moran's I 值为：{moran.I}")
        print(f"随机分布假设下Z检验值为：{moran.z_rand}")
        print(f"随机分布假设下Z检验的P值为：{moran.p_rand}")
        print(f"正态分布假设下Z检验值为：{moran.z_norm}")
        print(f"正态分布假设下Z检验的P值为：{moran.p_norm}")

        # print("绘制 Moran's I散点图...")
        # fig, ax = moran_scatterplot(moran, p=0.05, zstandard=True)
        # ax.set_xlabel('Variable')
        # ax.set_ylabel('Spatial Lag of Variable')
        # plt.show()

        # print("绘制 Moran's I指数图...")
        # fig, ax = plot_moran(moran, zstandard=True, figsize=(11, 6))
        # plt.show()

        # print("计算 Local Moran's I指数...")
        # moran_loc = Moran_Local(fieldValue, w)

        # print("绘制聚集区空间分布图...")
        # fig, ax = lisa_cluster(moran_loc, gdf, p=0.05, figsize=(10, 10), marker='.', markersize=5)
        # plt.show()

        # print("绘制显著性分布图...")
        # fig, ax = plot_local_Probably(moran_loc, gdf, fieldName, p=0.05)
        # plt.show()