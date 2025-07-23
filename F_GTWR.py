import numpy as np
import pandas as pd
import datetime as dt
import os
from time import time
import libpysal as ps
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import t
from mgtwr.model import GWR, MGWR, GTWR, MGTWR, GTWRF
from mgtwr.sel import SearchGWRParameter, SearchGTWRParameter, SearchMGWRParameter, SearchMGTWRParameter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from F_summary import summaryModel, summaryGLM, summaryGWR, summaryGTWR, summaryMGWR
import F_Regression as Regression
import gc


# from mgwr.utils import shift_colormap, truncate_colormap


def adj_alpha_(gtwr_results):
    """
    Corrected alpha (critical) values to account for multiple testing during hypothesis
    testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
    (.01) confidence levels. Correction comes from:

    :cite:`Silva:2016` : da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
    Geographically Weighted Regression. Geographical Analysis.

    """
    alpha = np.array([.1, .05, .001])
    pe = gtwr_results.ENP
    p = gtwr_results.k  # k: integer, Number of independent variables
    return (alpha * p) / pe


def critical_tval(gtwr_results, alpha=None):
    """
    Utility function to derive the critical t-value based on given alpha
    that are needed for hypothesis testing

    Parameters
    ----------
    alpha           : scalar
                      critical value to determine which tvalues are
                      associated with statistically significant parameter
                      estimates. Default to None in which case the adjusted
                      alpha value at the 95 percent CI is automatically
                      used.

    Returns
    -------
    critical        : scalar
                      critical t-val based on alpha
    """

    if alpha is not None:
        alpha = np.abs(alpha) / 2.0
        critical = t.ppf(1 - alpha, gtwr_results.n - 1)
    else:
        adj_alpha = adj_alpha_(gtwr_results)
        alpha = np.abs(adj_alpha[1]) / 2.0
        critical = t.ppf(1 - alpha, gtwr_results.n - 1)
    return critical


def filter_tvals(gtwr_results, critical_t=None, alpha=None):
    """
    Utility function to set tvalues with an absolute value smaller than the
    absolute value of the alpha (critical) value to 0. If critical_t
    is supplied than it is used directly to filter. If alpha is provided
    than the critical t value will be derived and used to filter. If neither
    are critical_t nor alpha are provided, an adjusted alpha at the 95
    percent CI will automatically be used to define the critical t-value and
    used to filter. If both critical_t and alpha are supplied then the alpha
    value will be ignored.

    Parameters
    ----------
    critical_t      : scalar
                      critical t-value to determine whether parameters are
                      statistically significant

    alpha           : scalar
                      alpha value to determine which tvalues are
                      associated with statistically significant parameter
                      estimates

    Returns
    -------
    filtered       : array
                      n*k; new set of n tvalues for each of k variables
                      where absolute tvalues less than the absolute value of
                      alpha have been set to 0.
    """
    if critical_t is not None:
        critical = critical_t
    else:
        critical = critical_tval(gtwr_results, alpha=alpha)

    subset = (gtwr_results.tvalues < critical) & (gtwr_results.tvalues > -1.0 * critical)
    tvalues = gtwr_results.tvalues.copy()
    tvalues[subset] = 0
    return tvalues


# 年（year）、1-365天数序号(dayNum)转换为日期类型
def out_date_by_day(year, dayNum):
    '''
    根据输入的年份和天数计算对应的日期
    '''
    first_day = dt.datetime.datetime(year, 1, 1)
    add_day = dt.datetime.timedelta(days=dayNum - 1)
    return dt.datetime.datetime.strftime(first_day + add_day, "%Y.%m.%d")


# 日期转换为（年*365）+day
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
    sum += (year - 2000) * 365 + day
    leap = 0
    # 接下来判断平年闰年
    if (year % 400 == 0) or ((year % 4) == 0) and (year % 100 != 0):  # and的优先级大于or
        # 1、世纪闰年:能被400整除的为世纪闰年
        # 2、普通闰年:能被4整除但不能被100整除的年份为普通闰年
        leap = 1
    if (leap == 1) and (month > 2):
        sum += 1  # 判断输入年的如果是闰年,且输入的月大于2月,则该年总天数加1
    return sum


# 日期转换为1-365天数序号
def out_day_by_date1(date):
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


def load_shape(shp_file):
    ######## 绘制分布概况
    georgia_shp = gp.read_file(shp_file)
    # ax = georgia_shp.plot(edgecolor='white', column="probabilit", cmap='GnBu', figsize=(6, 6))
    # georgia_shp.centroid.plot(ax=ax, color='r', marker='o')
    # ax.set_axis_off()

    return georgia_shp


def load_csv(csv_file, x_fields, y_field):
    print('加载文件数据：' + csv_file)
    geo_data = pd.read_csv(csv_file, low_memory=False)
    # print('数据概况\n', geo_data.head(5))

    # 因变量
    g_y = geo_data[y_field].values.reshape((-1, 1))
    # 自变量
    # g_X = geo_data[['YU_BI_DU', 'MEI_GQ_XJ', 'PINGJUN_XJ', 'V_GDP', 'V_POP',
    # 'V_SLOP', 'V_ASPE', 'V_NDVI', 'V_DEM', 'water_dist',
    # 'road_dist', 'V_lrad_', 'V_prec_', 'V_pres_', 'V_shum_',
    # 'V_srad_', 'V_temp_', 'V_wind_', 'V_tcl']].values
    g_X = geo_data[x_fields].values

    # # 坐标信息Latitude	Longitud
    # x = geo_data['X']
    # y = geo_data['Y']
    # t = geo_data['acq_date'].values.reshape((-1, 1))
    # g_coords = list(zip(x, y))
    # 坐标信息
    # x = geo_data['longitude'].values.reshape((-1, 1))
    # y = geo_data['latitude'].values.reshape((-1, 1))
    x = geo_data['X'].values.reshape((-1, 1)) - 12121089  # (70) #(80)12132277 # 12117861
    y = geo_data['Y'].values.reshape((-1, 1)) - 2835963  # (70) #(80)2835963 #2835228
    g_coords = np.hstack([x, y])

    # nt = []
    # t = geo_data['acq_date'].values.reshape((-1, 1))
    # for i in range(len(t)):
    # nt.append([out_day_by_date(dt.datetime.strptime(t[i][0], '%Y/%m/%d')) - 1])
    # nt = np.array(nt)
    nt = geo_data['date_num'].values.reshape((-1, 1))

    # ################ 数据标准化 ####################
    # 数据均值-方差标准化
    g_X = (g_X - g_X.mean(axis=0)) / g_X.std(axis=0)
    g_y = g_y.reshape((-1, 1))
    g_y = (g_y - g_y.mean(axis=0)) / g_y.std(axis=0)

    # # 数据标准化到0~1
    # g_X = (g_X - g_X.min(axis=0)) / (g_X.max(axis=0) - g_X.min(axis=0))
    # g_y = g_y.reshape((-1, 1))
    # g_y = (g_y - g_y.min(axis=0)) / (g_y.max(axis=0) - g_y.min(axis=0))

    # #scaler = StandardScaler()
    # #g_X = scaler.fit_transform(g_X)

    # scaler = MinMaxScaler()
    ## 对数据进行缩放处理
    # g_X = scaler.fit_transform(g_X)

    return g_coords, g_X, g_y, nt, geo_data


# numpy.linalg.LinAlgError: Matrix is singular.
# 奇异矩阵是一个常见以及棘手的问题，起因是距离过近带宽太小导致权重矩阵近似为0，可以尝试提高带宽的最小阈值
# 另一方面，对于模型的解释性使用AIC和AICc判定时，该值为回归模型间的相对值，需要和其他回归模型进行比较，
# 也即GTWR、GWR、OLS，更小的AIC或AICc表示模型更好。
######################## GWR ################################
def model_GWR(g_coords, g_X, g_y, fields):
    start = time()
    # print('GWR带宽选择...')
    # sel = SearchGWRParameter(g_coords, g_X, g_y, kernel='gaussian', fixed=True, thread=2)
    # # sel = SearchGWRParameter(g_coords, g_X, g_y, kernel='bisquare', fixed=False, max_iter=10)
    # bw = sel.search(criterion='AICc', verbose=True, time_cost=True, max_iter=10,
    # tol=1.0e-4)  # bw_min=9000, bw_max=13000, max_iter=10,
    bw = 9308.0
    # print('GWR最佳带宽bw：', bw)  # 80最佳带宽 12570.0  12682.0   # 70最佳带宽9308.0
    # exit(0)

    # gwr = GWR(g_coords, g_X, g_y, bw, kernel='bisquare', fixed=False)
    gwr = GWR(g_coords, g_X, g_y, bw, kernel='gaussian', fixed=True, thread=1)
    model_result = gwr.fit()
    #aa = gwr.predict(g_coords, g_X)
    end = time()
    print('计算耗时%f秒' % (end - start))

    # 预测效果验证
    Regression.regressor_Result(model_result.y, model_result.predict_value)
    Regression.draw_Result(model_result.y, model_result.predict_value, 'GWR')
    Regression.draw_RSM(g_X, g_y, fields, isScatter=True, isLine=True)  # 真实值拟合

    var_names = ['c_' + s for s in fields]
    var_names.insert(0, 'cof_Intercept')
    Regression.draw_RSM_individual(model_result.betas, model_result.predict_value, var_names, isScatter=True,
                                   isLine=False)  # 预测值拟合
    # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=True) 
    # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=True,degree=3)
    # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=True,degree=4)
    # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=True,degree=5)
    # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=True,degree=6)
    # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=True,degree=7)
    # Regression.draw_histogram(model_result.betas, var_names)

    print("score:", model_result.R2)
    summary = summaryModel(model_result) + summaryGWR(model_result)  # + summaryGLM(model_result)
    print(summary)
    return model_result


######################## MGWR ################################
def model_MGWR(g_coords, g_X, g_y, fields):
    start = time()
    print('MGWR带宽选择...')
    sel = SearchMGWRParameter(g_coords, g_X, g_y, kernel='gaussian', fixed=True, thread=5)
    # bws = sel.search(criterion='AICc', bw_min=9000, bw_max=13000, verbose=True)
    bws = sel.search(criterion='AICc', verbose=True, time_cost=True, tol=1.0e-3, tol_multi=1.0e-3, bws_same_times=2,
                     bw_min=9000, bw_max=10000)  # , multi_bw_min=[4], multi_bw_max=[12]
    print('MGWR最佳带宽bws：', bws)  # bws=12570.0

    mgwr = MGWR(g_coords, g_X, g_y, sel, kernel='gaussian', fixed=True, thread=4)
    # mgwr的fit方法可以传一个参数n_chunks分步计算，越大占用内存越小
    model_result = mgwr.fit(n_chunks=10000)

    end = time()
    print('计算耗时%f秒' % (end - start))

    # 预测效果验证
    Regression.regressor_Result(model_result.y, model_result.predict_value)
    Regression.draw_Result(model_result.y, model_result.predict_value, 'MGWR')
    Regression.draw_RSM(g_X, g_y, fields)

    var_names = ['c_' + s for s in fields]
    var_names.insert(0, 'cof_Intercept')
    Regression.draw_RSM(model_result.betas, g_y, var_names)
    Regression.draw_histogram(model_result.betas, var_names)

    print("score:", model_result.R2)
    summary = summaryModel(model_result) + summaryMGWR(model_result)
    print(summary)
    return model_result


####################### GTWR ################################
def model_GTWR(g_coords, nt, g_X, g_y, fields):
    start = time()
    # print('GTWR带宽选择...')
    # sel = SearchGTWRParameter(g_coords, nt, g_X, g_y, kernel='gaussian', fixed=True, thread=2)
    # bw, tau = sel.search(criterion='AICc', verbose=True, time_cost=True, bw_min=7200, bw_max=7800, tol=1.0e-4,
    # max_iter=10, tau_min=1.5, tau_max=5)
    # print('最佳带宽bw：', bw, '最佳时空尺度tau：', tau)
    # exit(0)
    #  70最佳带宽：bw:  7639.3 , tau:  0.2 , score:  27595.43293101382 ---bw:  7721.4 , tau:  0.2 , score:  27620.89332591382
    bw = 7639.3  # 7721.4  # 8307  # 12569.5
    tau = 0.2
    # gtwr = GTWR(g_coords, nt, g_X, g_y, bw, tau, kernel='bisquare', fixed=False)
    gtwr = GTWR(g_coords, nt, g_X, g_y, bw, tau, kernel='gaussian', fixed=True, thread=2)
    model_result = gtwr.fit()

    end = time()
    print('计算耗时%f秒' % (end - start))

    # 预测效果验证
    # Regression.regressor_importance(gtwr, g_X, g_y)
    Regression.regressor_Result(model_result.y, model_result.predict_value)
    Regression.draw_Result(model_result.y, model_result.predict_value, 'GTWR')

    print("score:", model_result.R2)
    summary = summaryModel(model_result) + summaryGTWR(model_result)
    print(summary)

    # Regression.draw_RSM(g_X, g_y, fields, isScatter=False, isLine=True)
    # Regression.draw_RSM(g_X, g_y, fields, isScatter=True, isLine=True)#真实值拟合

    var_names = ['c_' + s for s in fields]
    var_names.insert(0, 'cof_Intercept')
    # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=False)#预测值拟合   
    Regression.draw_RSM_individual(model_result.betas, model_result.predict_value, var_names, isScatter=True,
                                   isLine=False)  # 预测值拟合

    # Regression.draw_RSM(model_result.betas, g_y, var_names)
    Regression.draw_histogram(model_result.betas, var_names)
    return model_result


####################### GTWRF ################################
def model_GTWRF(g_coords, nt, g_X, g_y, fields):
    start = time()

    # print('GTWRF带宽选择...')
    # sel = SearchGTWRParameter(g_coords, nt, g_X, g_y, kernel='gaussian', fixed=True, thread=2)
    # bw, tau = sel.search(criterion='AICc', verbose=True, time_cost=True, bw_min=67200, bw_max=107800, tol=1.0e-4,
    #                      max_iter=10, tau_min=1.5, tau_max=10)
    # print('最佳带宽bw：', bw, '最佳时空尺度tau：', tau)
    # # # 70最佳带宽：bw:  7639.3 , tau:  0.2 , score:  27595.43293101382 ---bw:  7721.4 , tau:  0.2 , score:  27620.89332591382
    bw = 7639.3  # 7721.4  # 8307  # 12569.5
    tau = 0.2

    # print('搜索RandomForestRegressor模型最佳组合参数...')
    # # 参数值字典
    # parameters = {
    #     'n_estimators': [100, 200, 300],  # 树的数量
    #     'max_depth': [10, 12, 14],  # 树的最大深度
    #     'min_samples_split': [2, 4, 6],  # 节点分裂所需的最小样本数
    #     # 'min_samples_leaf': [10, 11, 12, 13, 14]
    # }
    # model = RandomForestRegressor()
    # grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='neg_mean_squared_error', cv=10,
    #                            n_jobs=2)
    # # 模型拟合
    # grid_result = grid_search.fit(g_X, g_y)
    # print("得分: %f 最佳组合的参数值 %s" % (grid_result.best_score_, grid_search.best_params_))
    # 得分: 0.567047 最佳组合的参数值 {'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 300}
    # 优化随机森林参数
    # best_rf_params = grid_search.best_params_
    # best_rf_params = {'max_depth': 12, 'min_samples_split': 4,
    # 'n_estimators': 300}
    best_rf_params = {'max_depth': 14, 'min_samples_split': 2,
                      'n_estimators': 300}

    # 训练GWRF模型
    gtwrf = GTWRF(g_coords, nt, g_X, g_y, bw=bw, tau=tau, rf_params=best_rf_params, kernel='gaussian',
                  fixed=True, min_neighbors=1, weight_threshold=1e-5, thread=10)
    # model_result = gtwrf.fit(X_train, y_train, coords_train, n_jobs=2)
    # model_result = gtwrf.fit(_fit_single_optimized=_fit_single_optimized)
    model_result = gtwrf.fit()

    # 对测试集进行预测
    # y_test_pred = gwrf.predict(X_train, y_train)

    end = time()
    print('计算耗时%f秒' % (end - start))

    # 预测效果验证
    Regression.draw_Result(model_result.y, model_result.predict_value, 'GTWRF')
    Regression.regressor_Result(model_result.y, model_result.predict_value)
    # Regression.regressor_importance(gtwr, g_X, g_y)

    print("score:", model_result.R2)
    summary = summaryModel(model_result) + summaryGTWR(model_result)
    print(summary)

    # var_names = ['c_' + s for s in fields]
    # # Regression.draw_RSM(model_result.betas, model_result.predict_value, var_names, isScatter=True, isLine=False)#预测值拟合
    # Regression.draw_RSM_individual(model_result.betas, model_result.predict_value, var_names, isScatter=True,
                                   # isLine=False)  # 预测值拟合

    # # Regression.draw_RSM(model_result.betas, g_y, var_names)
    # Regression.draw_histogram(model_result.betas, var_names)
    return model_result


####################### MGTWR ################################
def model_MGTWR(g_coords, nt, g_X, g_y, fields):
    start = time()
    print('MGTWR带宽选择...')
    sel = SearchMGTWRParameter(g_coords, nt, g_X, g_y, kernel='gaussian', fixed=True, thread=2)
    # bws = sel.search(multi_bw_min=[0.1], verbose=True, tol_multi=1.0e-4)
    bw, tau = sel.search(criterion='AICc', verbose=True, time_cost=True, tol=1.0e-4, tol_multi=1.0e-3, max_iter=5,
                         bw_min=10000, bw_max=15000, tau_min=1.5, tau_max=10)  # ,  tau_max=0.4

    print('最佳带宽bw：', bw, '最佳时空尺度tau：', tau)
    # mgtwr的fit方法可以传一个参数n_chunks分步计算，越大占用内存越小
    mgtwr = MGTWR(g_coords, nt, g_X, g_y, sel, kernel='gaussian', fixed=True, thread=2)
    model_result = mgtwr.fit(n_chunks=10000)

    end = time()
    print('计算耗时%f秒' % (end - start))

    # 预测效果验证
    Regression.regressor_Result(model_result.y, model_result.predict_value)
    Regression.draw_Result(model_result.y, model_result.predict_value, 'MGTWR')
    Regression.draw_RSM(g_X, g_y, fields)

    var_names = ['c_' + s for s in fields]
    var_names.insert(0, 'cof_Intercept')
    Regression.draw_RSM(model_result.betas, g_y, var_names)
    Regression.draw_histogram(model_result.betas, var_names)

    print("score:", model_result.R2)
    summary = summaryModel(model_result) + summaryGWR(model_result)
    print(summary)
    return model_result


def exportResult(shapeFile, model_result, geo_data, fields, Intercept=None):
    print('拟合参数空间化\n加入回归参数')

    var_names = ['cof_' + s for s in fields]
    if Intercept == None:
        var_names.insert(0, 'cof_Intercept')
    gwr_coefficent = pd.DataFrame(model_result.betas, columns=var_names)

    print('加入回归参数显著性')
    flter_names = ['p_' + s for s in fields]
    if Intercept == None:
        flter_names.insert(0, 'p_Intercept')

    # # gwr_flter_t = pd.DataFrame(np.abs(filter_tvals(model_result)), columns=flter_names)
    # gwr_flter_t = pd.DataFrame(np.abs(model_result.filter_tvals()), columns=flter_names) # 取出不限制的
    gwr_flter_t = pd.DataFrame(np.abs(model_result.tvalues), columns=flter_names)  # 原始t检验数据

    # w_names= ['w_' + s for s in fields]
    # gwr_w = pd.DataFrame(model_result.W, columns=w_names)

    print('加入预测值')
    predict_names = ['predict_value']
    # 数据标准化到0~1
    predict_value = (model_result.predict_value - model_result.predict_value.min(axis=0)) / (
            model_result.predict_value.max(axis=0) - model_result.predict_value.min(axis=0))
    gwr_predict = pd.DataFrame(predict_value, columns=predict_names)

    # 将点数据回归结果放到图层上展示
    # 考虑两个文件中的记录数可能不同，将矢量图层中的参加gwr的区域去掉
    georgia_data_geo = gp.GeoDataFrame(geo_data, geometry=gp.points_from_xy(geo_data.X, geo_data.Y), crs=3857)
    georgia_data_geo = georgia_data_geo.join(gwr_coefficent)
    georgia_data_geo = georgia_data_geo.join(gwr_flter_t)
    georgia_data_geo = georgia_data_geo.join(gwr_predict)

    georgia_data_geo.to_file(shapeFile)
    print('拟合参数空间化完成！')


def draw1(model_result, geo_data, georgia_shp, fields):
    ###########################################################
    # 原始值,预测值，系数，残差
    # Observed Predicte Intercept,c1,c2..., Residual StdError
    result = [model_result.y, model_result.predict_value, model_result.betas, model_result.std_res]
    print(result)

    # 拟合参数空间化，回归参数
    gwr_coefficent = pd.DataFrame(model_result.betas, columns=fields)

    print('回归参数显著性：')
    # gwr_flter_t = pd.DataFrame(filter_tvals(model_result))
    gwr_flter_t = pd.DataFrame(model_result.filter_tvals())

    # 将点数据回归结果放到面上展示
    # 主要是由于两个文件中的记录数不同，矢量面中的记录比csv中多几条，因此需要将没有参加gwr的区域去掉
    georgia_data_geo = gp.GeoDataFrame(geo_data, geometry=gp.points_from_xy(geo_data.X, geo_data.Y))
    georgia_data_geo = georgia_data_geo.join(gwr_coefficent)
    # 将回归参数与图层数据结合
    # georgia_shp_geo = gp.sjoin(georgia_shp, georgia_data_geo, how="inner", predicate='intersects').reset_index()  #
    georgia_shp_geo = gp.sjoin_nearest(georgia_shp, georgia_data_geo, how="inner").reset_index()
    print('计算完成！')

    print('绘制回归系数分布图...')
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))
    axes = ax.flatten()

    for i in range(0, len(fields) - 1):
        ax = axes[i]
        ax.set_title(fields[i])
        georgia_shp_geo.plot(ax=ax, column=fields[i], edgecolor='white', cmap='Blues', legend=True)

        if (gwr_flter_t[i] == 0).any():
            georgia_shp_geo[gwr_flter_t[i] == 0].plot(color='lightgrey', ax=ax, edgecolor='white')  # 灰色部分表示该系数不显著

        ax.set_axis_off()
        if i + 1 == 7:
            axes[7].axis('off')

    plt.show()


def draw(georgia_shp):
    # ###########################################################
    # # 原始值,预测值，系数，残差
    # # Observed Predicte Intercept,c1,c2..., Residual StdError
    # result = [model_result.y, model_result.predict_value, model_result.betas, model_result.std_res]
    # print(result)
    #
    # # 拟合参数空间化，回归参数
    # gwr_coefficent = pd.DataFrame(model_result.betas, columns=fields)
    #
    # print('回归参数显著性：')
    # # gwr_flter_t = pd.DataFrame(filter_tvals(model_result))
    # gwr_flter_t = pd.DataFrame(model_result.filter_tvals())
    #
    # # 将点数据回归结果放到面上展示
    # # 主要是由于两个文件中的记录数不同，矢量面中的记录比csv中多几条，因此需要将没有参加gwr的区域去掉
    # georgia_data_geo = gp.GeoDataFrame(geo_data, geometry=gp.points_from_xy(geo_data.X, geo_data.Y))
    # georgia_data_geo = georgia_data_geo.join(gwr_coefficent)
    # # 将回归参数与图层数据结合
    # # georgia_shp_geo = gp.sjoin(georgia_shp, georgia_data_geo, how="inner", predicate='intersects').reset_index()  #
    # georgia_shp_geo = gp.sjoin_nearest(georgia_shp, georgia_data_geo, how="inner").reset_index()
    # print('计算完成！')

    import contextily as ctx
    # var_names = ['cof_' + s for s in fields]
    var_names = ['cof_Interc', 'cof_YU_BI_', 'cof_PINGJU', 'cof_N_V_GD', 'cof_N_V_PO',
                 'cof_N_V_SL', 'cof_N_V_AS', 'cof_N_V_DE', 'cof_N_V_tc', 'cof_N_V_pr',
                 'cof_N_V_tm', 'cof_N_V_wi', 'cof_N_V_re', 'cof_N_wate', 'cof_N_road']
    # georgia_shp_geo = georgia_shp[var_names]
    # flter_names = ['p_' + s for s in fields]
    flter_names = ['p_Intercep', 'p_YU_BI_DU', 'p_PINGJUN_', 'p_N_V_GDP', 'p_N_V_POP',
                   'p_N_V_SLOP', 'p_N_V_ASPE', 'p_N_V_DEM', 'p_N_V_tcl', 'p_N_V_pre',
                   'p_N_V_tmp', 'p_N_V_wind', 'p_N_V_resi', 'p_N_water_', 'p_N_road_d']
    gwr_flter_t = georgia_shp[flter_names]

    print('绘制回归系数分布图...')
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(30, 20))
    axes = ax.flatten()

    for i in range(0, len(var_names) - 1):
        ax = axes[i]
        ax.set_title(var_names[i])

        georgia_shp.plot(ax=ax, column=var_names[i], cmap='Blues', legend=True, markersize=0.5)  # edgecolor='white',
        georgia_shp.centroid.plot(ax=ax, color='r', marker='.', markersize=0.5)

        if (gwr_flter_t[flter_names[i]] == 0).any():
            georgia_shp[gwr_flter_t[flter_names[i]] == 0].plot(color='lightgrey', ax=ax,
                                                               markersize=0.5)  # ,edgecolor='white' 灰色部分表示该系数不显著
        # ctx.add_basemap(ax,
        #                 # source='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
        #                 source="https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{x}/{y}",
        #                 zoom=11)
        ax.set_axis_off()

    axes[len(var_names)].axis('off')
    plt.show()


# 关联规则（Apriori）算法 # 要求数据值必须为True, False, 0, 1 ( the DataFrame are True, False, 0, 1)
def model_apriori(geo_data, fields, mlxtend=None, apriori=None):
    # pip install mlxtend
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
    from mlxtend.frequent_patterns import association_rules

    frequent_itemsets = apriori(geo_data[fields], min_support=0.5, use_colnames=True)
    ### alternatively:
    # frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
    # frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
    # frequent_itemsets = fpmax(df, min_support=0.6, use_colnames=True)

    # rules = association_rules(frequent_itemsets,num_itemsets=5, metric="confidence", min_threshold=0.7)
    rules = association_rules(frequent_itemsets, num_itemsets=6, metric='lift', min_threshold=1)
    print(rules)

    rules[(rules['lift'] > 1.125) & (rules['confidence'] > 0.8)]
    print(rules)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    print('\r', end='')

    data_dir = r'F:\Map\base'
    os.chdir(data_dir)

    yField = 'probability'  # area_Proba, Fire_Count, probability
    fields = ['N_V_tmp', 'N_V_pre', 'N_V_wind_',
              'N_V_SLOP', 'N_V_DEM', 'N_V_ASPE',
              'YU_BI_DU', 'PINGJUN_XJ', 'N_V_GDP', 'N_V_POP',
               'N_road_dist', 'N_water_dist'
              # 'N_railways',
              # 'N_V_wet','N_V_dtr','N_V_pet','N_V_frs', #'N_V_tmx','N_V_tmn',
              # 'N_V_vap','N_V_srad_','N_V_lrad_','N_V_shum_','N_V_prec_',#'N_V_temp_','N_V_pres_',
              # 'N_V_residence_density','N_V_NDVI'
              ]

    # georgia_shp = load_shape('GWR_Result_70.shp')
    # draw(georgia_shp)
    # exit(0)

    # 加载数据  
    g_coords, g_X, g_y, nt, geo_data = load_csv("modis_hn_Standardization.csv", fields, yField)

    g_X = g_X.astype(np.float32)  # 转换为更高效的内存数据类型
    g_y = g_y.astype(np.float32)  # 转换为更高效的内存数据类型
    nt = nt.astype(np.int32)  # 转换为更高效的内存数据类型
    g_coords = g_coords.astype(np.float32)

    # Month = geo_data['Month_'].values.reshape((-1, 1))
    # for i in range(1, 13):
    #     filtered_g_X = [a for a, b in zip(g_X, Month) if b == i]

    # # 关联规则
    # model_apriori(geo_data, fields)
    ################################################
    model_result = model_GWR(g_coords, g_X, g_y, fields)
    exportResult('GWR_Result.shp', model_result, geo_data, fields)

    # model_result = model_MGWR(g_coords, g_X, g_y, fields)
    # exportResult('MGWR_Result.shp', model_result, geo_data, fields)

    model_result = model_GTWR(g_coords, nt, g_X, g_y, fields)
    exportResult('GTWR_Result.shp', model_result, geo_data, fields)

    # model_result = model_GTWRF(g_coords, nt, g_X, g_y, fields)
    # exportResult('GTWRF_Result.shp', model_result, geo_data, fields, Intercept=1)

    # model_result = model_MGTWR(g_coords, nt, g_X, g_y, fields)
    # exportResult('MGTWR_Result.shp', model_result, geo_data, fields)

    # georgia_shp = load_shape('Output.shp')
    # draw(model_result,geo_data,georgia_shp,fields)
