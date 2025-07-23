# -*- coding: utf-8 -*-
# 异常值分析
# 异常值是指样本中的个别值，其数值明显偏离其余的观测值。
# 异常值也称离群点，异常值的分析也称为离群点的分析。

# 异常值分析 → 3σ原则 / 箱型图分析
# 异常值处理方法 → 删除 / 修正填补
#############################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_csv(csv_file):
    print('加载文件数据：' + csv_file)
    geo_data = pd.read_csv(csv_file, low_memory=False)
    print(geo_data.head())
    return geo_data


# 3σ原则 -未调试
def Outlier_3σ(g_X):
    # （1）3σ原则：如果数据服从正态分布，异常值被定义为一组测定值中与平均值的偏差超过3倍的值 → p(|x - μ| > 3σ) ≤ 0.003
    u = g_X.mean()  # 计算均值
    std = g_X.std()  # 计算标准差
    stats.kstest(g_X, 'norm', (
        u, std))  # 正态分布的方式，得到 KstestResult(statistic=0.012627414595288711, pvalue=0.082417721086262413)，P值>0.5
    print('均值为：%.3f，标准差为：%.3f' % (u, std))
    print('------')

    # 正态性检验
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.plot(kind='kde', grid=True, style='-k', title='密度曲线')
    plt.axvline(3 * std, color='r', linestyle="--", alpha=0.8)  # 3倍的标准差
    plt.axvline(-3 * std, color='r', linestyle="--", alpha=0.8)

    # 绘制数据密度曲线
    error = g_X[np.abs(g_X - u) > 3 * std]  # 超过3倍差的数据（即异常值）筛选出来
    data_c = g_X[np.abs(g_X - u) < 3 * std]
    print('异常值共%i条' % len(error))
    ax2 = fig.add_subplot(2, 1, 2)

    # 图表表达
    plt.scatter(data_c, data_c, color='k', marker='.', alpha=0.3)
    plt.scatter(error, error, color='r', marker='.', alpha=0.7)
    plt.xlim([-10, 10010])
    plt.grid()


# 箱型图分析  -未调试
def Outlier_box(g_X):
    # （2）箱型图分析
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    plt.plot(vert=False, grid=True, color=color, ax=ax1, label='样本数据')
    # 箱型图看数据分布情况
    # 以内限为界
    s = g_X.describe()
    print(s)
    print('------')
    # 基本统计量

    q1 = s['25%']
    q3 = s['75%']
    iqr = q3 - q1
    mi = q1 - 1.5 * iqr
    ma = q3 + 1.5 * iqr
    print('分位差为：%.3f，下限为：%.3f，上限为：%.3f' % (iqr, mi, ma))
    print('------')
    # 计算分位差

    ax2 = fig.add_subplot(2, 1, 2)
    error = g_X[(g_X < mi) | (g_X > ma)]
    data_c = g_X[(g_X >= mi) & (g_X <= ma)]
    print('异常值共%i条' % len(error))
    # 筛选出异常值error、剔除异常值之后的数据data_c

    plt.scatter(data_c.index, data_c, color='k', marker='.', alpha=0.3)
    plt.scatter(error.index, error, color='r', marker='.', alpha=0.5)
    plt.xlim([-10, 10010])
    plt.grid()


# 定义箱线图识别异常值函数
def box_mean(X, Low=None, Up=None):
    # 对给定的某列Series进行异常值检测，并用不包括异常值的剩余数据的均值、最小值、最大值替换（X：进行异常值分析的DataFrame的某一列）
    if Low is None:
        # Low=X.quantile(0.25)-1.5*(X.quantile(0.75)-X.quantile(0.25))
        Low = X.mean() - 3 * X.std()
    if Up is None:
        # Up=X.quantile(0.75)+1.5*(X.quantile(0.75)-X.quantile(0.25))
        Up = X.mean() + 3 * X.std()

    X_select = X[(X >= Low) & (X <= Up)]  # 取出不包含异常点的数据，为了求均值、最小值、最大值
    X_new = X[:]
    # X[(X<Low) | (X>Up)]=X_select.mean()  #用非异常数据的那些数据的均值、最小值、最大值替换异常值
    X_new[(X_new < Low)] = X_select.min()
    X_new[(X_new > Up)] = X_select.max()

    return Low, Up, X_new


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    print('\r', end='')
    data_dir = r'F:\Map\base'
    os.chdir(data_dir)

    # 读取数据
    data = load_csv("modis_hn_Standardization.csv") 

    cols = ['V_GDP', 'V_POP', 'V_SLOP', 'V_ASPE', 'V_NDVI', 'V_DEM', 
            'water_dist', 'railways_d', 'road_dist', 'V_dtr', 'V_frs', 'V_pet', 'V_pre', 'V_tmn',
            'V_tmp', 'V_tmx', 'V_vap', 'V_wet', 'V_lrad_', 'V_prec_', 'V_pres_', 'V_shum_',
            'V_srad_', 'V_temp_', 'V_wind_', 'V_water_dist', 'V_road_dist', 
            'V_residence_density']
    data1 = pd.DataFrame(data=[])  # , columns=cols
    for col_name in cols:
        Low, Up, data1['N_' + col_name] = box_mean(data[col_name])
        print(col_name, ':最小值', data[col_name].min(),
              ',最大值', data[col_name].max(),
              ',均值', data[col_name].mean(),
              ',均方差', data[col_name].std(),
              ']\t[', np.around(Low, 2), ',', np.around(Up, 2), ']')

    # data_new = list(zip(data, data1))
    data_new = [data, data1]
    df = pd.concat(data_new, axis=1)
    # df = pd.DataFrame(data=data_new, columns=cols)
    df.to_csv("modis_hn_80_Standardization.csv", index=0, sep=',')

    # col_numbers=len(cols)    #数值型特征列的个数
    # pic_numbers_per_line=2 #每行显示的图形个数
    # row_numbers=col_numbers//pic_numbers_per_line   #总共显示几行图形
    # plt.figure(figsize=(20,16),dpi=100) 
    # for i in range(col_numbers):
    # plt.subplot(row_numbers,pic_numbers_per_line,i+1)
    # plt.boxplot(data[cols[i]])
    # plt.show()

    ##删除DataFrame中某列有异常值的整行
    # for col_name in outlier_cols:
    #    yczindex_list=outlier_index(selfdata[col_name])  
    #    selfdata=selfdata.drop(yczindex_list,axis=0)

# #数据归一化/ 标准化
# #数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间。
# #在某些比较和评价的指标处理中经常会用到，去除数据的单位限制，将其转化为无量纲的纯数值，便于不同单位或量级的指标能够进行比较和加权
# #最典型的就是数据的归一化处理，即将数据统一映射到[0,1]区间上 
# # 在分类、聚类算法中，需要使用距离来度量相似性的时候，Z-score表现更好 

# #（1）0 - 1标准化
# # 将数据的最大最小值记录下来，并通过Max-Min作为基数（即Min=0，Max=1）进行数据的归一化处理
# #x = (x - Min) / (Max - Min)
# def data_norm(df, *cols):
# df_n = df.copy()
# for col in cols:
# d_max = df_n[col].max()
# d_min = df_n[col].min()
# df_n[col + '_n'] = (df_n[col] - d_min) / (d_max - d_min)
# return(df_n)

# # 创建函数，标准化数据
# df_n = data_norm(df, 'value1',  'value2')


# #（2）Z - score标准化
# # Z分数（z-score）,是一个分数与平均数的差再除以标准差的过程 → z=(x-μ)/σ，其中x为某一具体分数，μ为平均数，σ为标准差
# # Z值的量代表着原始分数和母体平均值之间的距离，是以标准差为单位计算。在原始分数低于平均值时Z则为负数，反之则为正数
# # 数学意义：一个给定分数距离平均数多少个标准差?
# def data_Znorm(df, *cols):
# df_n = df.copy()
# for col in cols:
# u = df_n[col].mean()
# std = df_n[col].std()
# df_n[col + '_Zn'] = (df_n[col] - u) / std #平均值/标准差
# return(df_n)

# # 创建函数，标准化数据
# df_z = data_Znorm(df,'value1','value2')
# u_z = df_z['value1_Zn'].mean()
# std_z = df_z['value1_Zn'].std()
# print(df_z)
# print('标准化后value1的均值为:%.2f, 标准差为：%.2f' % (u_z, std_z))
# # 经过处理的数据符合标准正态分布，即均值为0，标准差为1
