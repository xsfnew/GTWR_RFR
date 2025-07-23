import os
from collections.abc import Iterable
from time import time
import pandas as pd
import geopandas as gpd
import libpysal
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
from esda.moran import Moran
from splot.esda import moran_scatterplot
import warnings
import os
from sklearn.model_selection import train_test_split, learning_curve, validation_curve  # 将数据集分开成训练集和测试集
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lars, LarsCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss, zero_one_loss, classification_report
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
import seaborn as sns  # matplotlib的高级API
import scipy.stats as stats
import collections


def model_DecisionTreeClassifier(X_train, y_train, X_test, y_test):
    # 将各参数值以字典形式组织起来
    parameters = {
        'max_depth': [2, 3, 4, 5, 6],  # max_depth = [18, 19, 20, 21, 22]
        'min_samples_split': [2, 4, 6, 8],  # min_samples_split = [2, 4, 6, 8]
        'min_samples_leaf': [2, 4, 8, 10, 12]  # min_samples_leaf = [2, 4, 8]
    }
    model = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=10, n_jobs=2)
    # 模型拟合
    grid_result = grid_search.fit(X_train, y_train)
    print("得分: %f 最佳组合的参数值 %s" % (grid_result.best_score_, grid_search.best_params_))

    # 构建分类决策树
    CART_Class = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=4)
    # 模型拟合
    decision_tree = CART_Class.fit(X_train, y_train)
    # 模型在测试集上的预测
    pred = CART_Class.predict(X_test)
    # 模型的准确率
    print('模型在测试集的预测准确率：\n', metrics.accuracy_score(y_test, pred))
    # 计算衡量模型好坏的MSE值
    metrics.mean_squared_error(y_test, pred)
    draw_auc(CART_Class, X_test, y_test)


# # 需要在电脑中安装Graphviz
# # https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# # 然后将解压文件中的bin设置到环境变量中
# # 导入第三方模块
# from sklearn.tree import export_graphviz
# from IPython.display import Image
# import pydotplus
# from sklearn.externals.six import StringIO
#
# # 绘制决策树
# dot_data = StringIO()
# export_graphviz(
#     decision_tree,
#     out_file=dot_data,
#     feature_names=predictors,
#     class_names=['Unsurvived', 'Survived'],
#     # filled=True,
#     rounded=True,
#     special_characters=True
# )
# # 决策树展现
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

def model_RandomForestClassifier(X_train, y_train, X_test, y_test):
    # 构建随机森林
    model = ensemble.RandomForestClassifier(n_estimators=200, random_state=1234)
    # 随机森林的拟合
    model.fit(X_train, y_train.astype('int'))
    # 模型在测试集上的预测
    RF_pred = model.predict(X_test)
    # 模型的准确率
    print('模型在测试集的预测准确率：', metrics.accuracy_score(y_test.astype('int'), RF_pred))

    # # 计算绘图数据
    # # ROC曲线是根据一系列不同的二分类方式（分界值或决定阈），以真阳性率（灵敏度）为纵坐标，假阳性率（1-特异度）为横坐标绘制的曲线。
    # # 传统的诊断试验评价方法有一个共同的特点，必须将试验结果分为两类，再进行统计分析。
    # y_score = model.predict_proba(X_test)[:, 1]
    # fpr, tpr, threshold = metrics.roc_curve(y_test.astype('int'), y_score)
    # roc_auc = metrics.auc(fpr, tpr)
    # print('模型roc_auc：', roc_auc)
    #
    # # 计算模型的MSE值
    # metrics.mean_squared_error(y_test.astype('int'), RF_pred)
    #
    # # 绘图
    # plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    # plt.plot(fpr, tpr, color='black', lw=1)
    # plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)
    # plt.xlabel('1-Specificity')
    # plt.ylabel('Sensitivity')
    # plt.show()

    # 构建变量重要性的序列
    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    # 排序并绘图
    # importance.sort_values().plot('barh')
    importance.sort_values(ascending=True).plot('YU_BI_DU')
    plt.show()


##################################################################################
def model_LogisticRegression(X_train, y_train, X_test, y_test):
    # # 请尝试将L1正则和L2正则分开，并配合合适的优化求解算法（slover）
    # # parameters = {'penalty':['l1','l2'],
    # #                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # #              }
    # ## 优化模型,选择模型最佳参数
    # # parameters = {'penalty': ('l1', 'l2'), 'C': (0.01, 0.1, 1, 10)}
    # penaltys = ['l1', 'l2']
    # Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # parameters = dict(penalty=penaltys, C=Cs)
    # model = LogisticRegression()
    # grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='r2', n_jobs=2,
    #                            cv=10)  # , scoring='neg_mean_squared_error'
    # grid_search.fit(X_train, y_train)
    # best_parameters = grid_search.best_estimator_.get_params()
    # print('所有最佳参数：n', best_parameters)
    # print('最佳参数：n', grid_search.best_params_)
    # print('最佳效果得分：%0.3f' % grid_search.best_score_)
    #
    # # 交叉验证用于评估模型性能和进行参数调优（模型选择）
    # # 分类任务中交叉验证缺省是采用StratifiedKFold
    # loss = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_log_loss')
    # print('logloss of each fold is: ', -loss)
    # print('cv logloss is:', -loss.mean())
    #
    # grid_search.cv_results_
    # # 绘制plot CV误差曲线
    # test_means = grid_search.cv_results_['mean_test_score']
    # test_stds = grid_search.cv_results_['std_test_score']
    # # train_means = grid_search.cv_results_['mean_train_score']
    # # train_stds = grid_search.cv_results_['std_train_score']
    #
    # # plot results
    # n_Cs = len(Cs)
    # number_penaltys = len(penaltys)
    # test_scores = np.array(test_means).reshape(n_Cs, number_penaltys)
    # test_stds = np.array(test_stds).reshape(n_Cs, number_penaltys)
    # # train_scores = np.array(train_means).reshape(n_Cs, number_penaltys)
    # # train_stds = np.array(train_stds).reshape(n_Cs, number_penaltys)
    #
    # x_axis = np.log10(Cs)
    # for i, value in enumerate(penaltys):
    #     # pyplot.plot(log(Cs), test_scores[i], label= 'penalty:'   + str(value))
    #     plt.errorbar(x_axis, test_scores[:, i], yerr=test_stds[:, i], label=penaltys[i] + ' Test')
    #     # plt.errorbar(x_axis, train_scores[:, i], yerr=train_stds[:, i], label=penaltys[i] + ' Train')
    #
    # plt.legend()
    # plt.xlabel('log(C)')
    # plt.ylabel('neg-logloss')
    # # plt.savefig('LogisticGridSearchCV_C.png')
    # plt.show()

    # 训练模型
    logisticregression = LogisticRegression(penalty='l2', C=1)
    # logisticregression = LogisticRegression(grid_search.best_params_)
    model = logisticregression.fit(X_train, y_train)

    # 测试模型
    y_predict_test = model.predict(X_test)  # 真实值与预测值

    # 预测
    w = model.coef_  # 得到权重列表
    b = model.intercept_  # 得到bias值
    # print('解释变量、系数（保留5位小数）:\n', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (b))
    print([model.intercept_] + model.coef_.tolist())
    # for i in range(0, len(w)):
    #     print(w[i])
    # print('获取概率值', '-' * 30, 'n', model.predict_proba(y_test))

    y_predict_prob = model.predict_proba(X_test)  # 预测概率
    y_predict_df = pd.DataFrame(y_predict_test, columns=['y_predict'])  # , index=y_test.index)
    y_test_predict_df = pd.concat([y_test, y_predict_df, y_predict_prob], axis=1)
    print('真实值、预测值、概率值', '-' * 30, 'n', y_test_predict_df)

    z = w + b * X_test
    # 将z值带入逻辑回归函数中，得到概率值
    y_pred = 1 / (1 + np.exp(-z))
    print('预测的概率值：', y_pred)

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'Linear')
    draw_learning_curve(model, X_train, y_train)

    # 模型优度的可视化展现
    fpr, tpr, _ = metrics.roc_curve(y_test, y_predict_test, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    plt.style.use('ggplot')  # 设置绘图风格
    plt.plot(fpr, tpr, '')  # 绘制ROC曲线
    plt.plot((0, 1), (0, 1), 'r--')  # 绘制参考线
    plt.text(0.5, 0.5, 'AUC=%.2f' % auc)  # 添加文本注释
    plt.title('LogisticRegression ROC')  # 设置标题
    plt.xlabel('False Positive Rate')  # 设置坐标轴标签
    plt.ylabel('True Positive Rate')
    plt.tick_params(top='off', right='off')  # 去除图形顶部边界和右边界的刻度
    plt.show()  # 图形显示


# 岭回归是一种分析多重共线性的多元回归的技术。岭回归也称为吉洪诺夫正则化。
def model_Ridge(X_train, y_train, X_test, y_test, fields):
    print('岭回归模型')
    # 构造不同的alpha值
    # alpha值越高，对系数的限制越多;
    # alpha低则对系数几乎没有限制，其泛化能力更强，在这种情况下，线性回归和岭回归类似。
    # 欠拟合，则降低alpha值
    # 过拟合，则增加alpha值
    # 创建不同的正则化参数
    # alphas = np.logspace(-6, 6, 13)  # 从10的-6次方到10的6次方，共13个参数
    alphas = np.logspace(1, 10, 50)
    # #alphas = np.logspace(-1, 3, 100)

    # # 初始化用于存储R-squared得分的列表
    # scores = []

    # # 在不同正则化参数下训练模型并计算得分
    # for alpha in alphas:
    # ridge_cv = RidgeCV(alphas=[alpha], cv=5)
    # ridge_cv.fit(X_train, y_train)
    # score = ridge_cv.score(X_test, y_test)
    # scores.append(score)

    # # 绘制正则化参数与R-squared得分的关系
    # plt.figure(figsize=(10, 6))
    # plt.semilogx(alphas, scores, marker='o')
    # plt.xlabel('Regularization Parameter (alpha)')
    # plt.ylabel('R-squared Score')
    # plt.title('Impact of Regularization on RidgeCV Performance')
    # plt.grid(True)
    # plt.show()

    # 岭回归模型的交叉验证
    # 设置交叉验证的参数，对于每一个Lambda值，都执行10重交叉验证
    ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=100)
    # 模型拟合
    ridge_cv.fit(X_train, y_train)
    best_alpha = ridge_cv.alpha_
    print("最佳的alpha值:%f 最佳的alpha得分:%f" % (best_alpha, ridge_cv.score(X_train, y_train)))

    # 基于最佳的alpha值建模
    # Ridge参数：
    # alpha：正则化系数，float类型，默认为1.0。正则化改善了问题的条件并减少了估计的方差。较大的值指定较强的正则化。
    # fit_intercept：是否需要截距，bool类型，默认为True。也就是是否求解b。
    # normalize：是否先进行归一化，bool类型，默认为False。如果为真，则回归X将在回归之前被归一化。 当fit_intercept设置为False时，将忽略此参数。 当回归量归一化时，注意到这使得超参数学习更加鲁棒，并且几乎不依赖于样本的数量。 相同的属性对标准化数据无效。然而，如果你想标准化，请在调用normalize = False训练估计器之前，使用preprocessing.StandardScaler处理数据。
    # copy_X：是否复制X数组，bool类型，默认为True，如果为True，将复制X数组; 否则，它覆盖原数组X。
    # max_iter：最大的迭代次数，int类型，默认为None，最大的迭代次数，对于sparse_cg和lsqr而言，默认次数取决于scipy.sparse.linalg，对于sag而言，则默认为1000次。
    # tol：精度，float类型，默认为0 .001。就是解的精度。
    # solver：求解方法，str类型，默认为auto。可选参数为：auto、svd、cholesky、lsqr、sparse_cg、sag。
    #   auto根据数据类型自动选择求解器。
    #   svd使用X的奇异值分解来计算Ridge系数。对于奇异矩阵比cholesky更稳定。
    #   cholesky使用标准的scipy.linalg.solve函数来获得闭合形式的解。
    #   sparse_cg使用在scipy.sparse.linalg.cg中找到的共轭梯度求解器。作为迭代算法，这个求解器比大规模数据（设置tol和max_iter的可能性）的cholesky更合适。
    #   lsqr使用专用的正则化最小二乘常数scipy.sparse.linalg.lsqr。它是最快的，但可能在旧的scipy版本不可用。它是使用迭代过程。
    #   sag使用随机平均梯度下降。它也使用迭代过程，并且当n_samples和n_feature都很大时，通常比其他求解器更快。注意，sag快速收敛仅在具有近似相同尺度的特征上被保证。您可以使用sklearn.preprocessing的缩放器预处理数据。
    # random_state：sag的伪随机种子。
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    # 返回岭回归系数
    # pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[model.intercept_] + model.coef_.tolist())
    # 预测
    y_predict_test = model.predict(X_test)

    w = model.coef_  # 得到权重列表
    b = model.intercept_  # 得到bias值
    # print('ridge输出解释变量、系数（保留5位小数）:', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (b))
    print([model.intercept_] + model.coef_.tolist())
    # for i in range(0, len(w)):
    #     print(w[i])

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'Ridge')


# 套索回归(Least Absolute Shrinkage and Selection Operator)是一种既进行变量选择又进行正则化的方法，提高其生成的统计模型的预测精度和可解释性。
def model_Lasso(X_train, y_train, X_test, y_test, fields):
    print('套索回归模型')
    # 构造不同的Lambda值
    Lambdas = np.logspace(-5, 5, 50)
    # # 构造空列表，用于存储模型的偏回归系数
    # lasso_cofficients = []
    # for Lambda in Lambdas:
    #     model = Lasso(alpha=Lambda, max_iter=10000)
    #     model.fit(X_train, y_train)
    #     lasso_cofficients.append(model.coef_)
    #
    # # 绘制Lambda与回归系数的关系
    # plt.plot(Lambdas, lasso_cofficients)
    # # 对x轴作对数变换
    # plt.xscale('log')
    # # 设置折线图x轴和y轴标签
    # plt.xlabel('Lambda')
    # plt.ylabel('Cofficients')
    # # 显示图形
    # plt.show()

    # LASSO回归模型的交叉验证
    lasso_cv = LassoCV(alphas=Lambdas, cv=50, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    # 输出最佳的alpha值
    best_alpha = lasso_cv.alpha_
    print('Lasso输出解释变量最佳的alpha值', best_alpha)

    # 基于最佳的alpha值建模
    model = Lasso(alpha=best_alpha, max_iter=10000)
    model.fit(X_train, y_train)
    # 返回LASSO回归的系数
    # pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[model.intercept_] + model.coef_.tolist())

    # 预测
    y_predict_test = model.predict(X_test)
    w = model.coef_  # 得到权重列表
    b = model.intercept_  # 得到bias值
    # print('Lasso输出解释变量、系数（保留5位小数）:', model.intercept_, [np.around(i, 5) for i in model.coef_])
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    print([model.intercept_] + model.coef_.tolist())

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'Lasso')


# 弹性网络回归（ElasticNet Regression）是对岭回归和Lasso回归的融合，其惩罚项是对L1范数和L2范数的一个权衡。先将有共线性的自变量分成一组，如果其中有一个自变量与因变量有强相关关系，那么就将这一组所有自变量都输入线性模型。
def model_ElasticNet(X_train, y_train, X_test, y_test, fields):
    print('弹性网络回归模型')
    # 构造不同的Lambda值
    Lambdas = np.logspace(-10, 10, 50)
    l1_ratios = np.linspace(0, 1, 10)

    # 模型的交叉验证
    enet_cv = ElasticNetCV(alphas=Lambdas, l1_ratio=l1_ratios, cv=50, max_iter=10000)
    enet_cv.fit(X_train, y_train)
    # 输出最佳的alpha值
    best_alpha = enet_cv.alpha_
    best_l1_ratio = enet_cv.l1_ratio_
    print('ElasticNet输出解释变量最佳的alpha值：', best_alpha, '最佳的l1_ratio值：', best_l1_ratio)

    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000)
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_test)
    w = model.coef_  # 得到权重列表
    b = model.intercept_  # 得到bias值
    # print('ElasticNet输出解释变量、系数（保留5位小数）:', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    print([model.intercept_] + model.coef_.tolist())

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'ElasticNet')


# 贝叶斯岭回归(Bayesian Ridge Regression)在最大似然估计中很难决定模型的复杂程度，
# Ridge回归加入的惩罚参数其实也是解决这个问题的，同时可以采用的方法还有对数据进行正规化处理，
# 另一个可以解决此问题的方法就是采用贝叶斯方法。
def model_BayesianRidge(X_train, y_train, X_test, y_test, fields):
    print('贝叶斯岭回归模型')
    model = BayesianRidge(compute_score=True)
    model.fit(X_train, y_train)
    w = model.coef_
    b = model.intercept_  # 得到bias值
    y_predict_test = model.predict(X_test)
    # print('BayesianRidge输出解释变量、系数（保留5位小数）:', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    print([model.intercept_] + model.coef_.tolist())

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'BayesianRidge')


# 最小回归角回归(Least-Angled Regression (LARS))算法
# 由于LARS的迭代方向是根据目标的残差而定，所以该算法对样本的噪声极为敏感。
def model_Lars(X_train, y_train, X_test, y_test, fields):
    print('最小回归角回归模型')

    # 模型的交叉验证
    lars_cv = LarsCV(cv=10, max_iter=10000)
    lars_cv.fit(X_train, y_train)
    # 输出最佳的alpha值
    best_alpha = lars_cv.alpha_
    print('LarsCV最佳的alpha值：', best_alpha)

    # model = LassoLars(alpha=.1)
    model = LassoLars(alpha=best_alpha)
    model.fit(X_train, y_train)
    w = model.coef_
    b = model.intercept_  # 得到bias值
    y_predict_test = model.predict(X_test)
    # print('LarsCV输出解释变量、系数（保留5位小数）:', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    print([model.intercept_] + model.coef_.tolist())

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'LassoLars')


def model_LassoLars(X_train, y_train, X_test, y_test, fields):
    print('LassoLars回归模型')
    # 模型的交叉验证
    lars_cv = LassoLarsCV(cv=5)
    lars_cv.fit(X_train, y_train)
    # 输出最佳的alpha值
    best_alpha = lars_cv.alpha_
    print('LassoLarsCV最佳的alpha值：', best_alpha)

    model = LassoLars(alpha=best_alpha)
    model.fit(X_train, y_train)
    w = model.coef_
    b = model.intercept_  # 得到bias值
    y_predict_test = model.predict(X_test)
    # print('LassoLarsCV输出解释变量、系数（保留5位小数）:', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    print([model.intercept_] + model.coef_.tolist())

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'LassoLars')


# 偏最小二乘回归（Partial least squares regression， PLS回归）是一种统计学方法，与主成分回归有关系，
# 但不是寻找响应和独立变量之间最小方差的超平面，而是通过投影预测变量和观测变量到一个新空间来寻找一个线性回归模型。
def model_PLSRegression(X_train, y_train, X_test, y_test, fields):
    print('偏最小二乘回归模型')
    model = PLSRegression(n_components=4)  # 要保留的主成分数，默认为2个。
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_test)
    w = model.coef_
    b = model.intercept_  # 得到bias值
    # print('PLSRegression输出解释变量、系数（保留5位小数）:', b, [np.around(i, 5) for i in w[0]])
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    print([model.intercept_] + model.coef_.tolist())

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'PLS')


# 多元线性回归
def model_LinearRegression(X_train, y_train, X_test, y_test, fields):
    print('线性回归模型')
    # n_jobs: 默认None (相当于1)，使用多少个processor完成这个拟合任务，数据量较大且CPU性能较好且有多核情况下可以使用-1这个参数，调用所有processor计算，减少运算时间。
    # copy_X: 默认True，特征矩阵X是否需要拷贝，拷贝一份scikit-learn做的运算不影响我们的原始数据，否则X矩阵有可能会被覆盖。一般使用 True
    # fit_intercept: 默认True，模型是否拟合截距项w0。
    model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
    model.fit(X_train, y_train)

    # 预测
    y_predict_test = model.predict(X_train)
    Residual(y_train, y_predict_test)
    exit(0)
    w = model.coef_  # 得到权重列表
    b = model.intercept_  # 得到bias值
    # print('解释变量、系数（保留5位小数）:\n', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (b))
    print([model.intercept_] + model.coef_.tolist())
    # for i in range(0, len(w)):
    #     print(w[i])

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'Linear Regression')

    # draw_histogram(X_test, fields) 
    # draw_RSM(X_test, y_predict_test, fields)
    # regressor_importance(model, X_train, y_train, fields)
    # draw_learning_curve(model, X_train, y_train)


# 多项式回归
def model_polyfit(X_train, y_train, X_test, y_test, fields, degree=4):
    from sklearn.preprocessing import PolynomialFeatures
    print('多项式回归PolynomialFeatures模型')
    # 多项式回归
    poly = PolynomialFeatures(degree=degree)  # 设置最多几次幂
    # poly.fit(X_train)
    # X_train_new = poly.transform(X_train)
    X_train_new = poly.fit_transform(X_train)
    # 这个多项式回归是对x进行处理后，让其成为非线性关系，如：
    # print(X_train.shape)
    # print(X_train_new.shape)
    # # (15026, 14)
    # # (15026, 120)
    # 之后的操作与LR完全相同，所以Polynomial并没有作为独立的API，而是放在preprocessing

    model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
    model.fit(X_train_new, y_train)
    X_test_new = poly.transform(X_test)
    # 预测
    y_predict_test = model.predict(X_test_new)
    b = model.intercept_  # 得到bias值
    w = model.coef_  # 得到权重列表

    # print('解释变量、系数（保留5位小数）:\n', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (b))
    print([model.intercept_] + model.coef_.tolist())
    # for i in range(0, len(w)):
    #     print(w[i])

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train_new, y_train), 5)))
    # regressor_importance(model, X_train, y_train)## 特征项不符合
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'Polynomial')


##################################################################################
def model_KNeighborsRegressor(X_train, y_train, X_test, y_test, fields):
    print('KNeighborsRegressor模型')
    # score = []
    # alphas = []
    # for alpha in range(1, 20, 1):
    # alphas.append(alpha)
    # model = KNeighborsRegressor(alpha)
    # sc = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10))
    # s = sc.mean()
    # score.append(s)
    # print('alpha:%d' % alpha + ' score:%s' % s)
    # plt.plot(alphas, score)
    # plt.show()
    # exit()

    model = KNeighborsRegressor(n_neighbors=14)  # 默认的K邻域为12
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_train)
    Residual(y_train, y_predict_test)
    return

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'KNeighbors')


# 决策树回归模型(DecisionTreeRegressor算法)
def model_DecisionTreeRegressor(X_train, y_train, X_test, y_test, fields):
    print('DecisionTreeRegressor模型')
    # print('搜索最佳组合参数...')
    # parameters = {
    # 'max_depth': [8, 9, 10, 11, 12],
    # 'min_samples_split': [2, 3, 4, 6, 8, 10],
    # 'min_samples_leaf': [10, 11, 12, 13, 14]
    # }
    # model = DecisionTreeRegressor(random_state=42, splitter='best')
    # grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='r2', cv=10, n_jobs=2)#, scoring='neg_mean_squared_error'
    # grid_result = grid_search.fit(X_train, y_train)
    # print("得分: %f 最佳组合的参数值 %s" % (grid_result.best_score_, grid_search.best_params_))
    # return
    # # 得分: 0.411426    最佳组合的参数值    {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 2}

    model = DecisionTreeRegressor(random_state=42, splitter='best', max_depth=10, min_samples_leaf=11,
                                  min_samples_split=2)  # ,max_features='log2'
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_test)

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 4)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'DecisionTree')


def model_RandomForestRegressor(X_train, y_train, X_test, y_test, fields):
    print('RandomForestRegressor模型')
    # print('搜索最佳组合参数...')
    # # 参数值字典
    # parameters = {
    # 'n_estimators': [100, 200, 300],  # 树的数量
    # 'max_depth': [10, 12, 14],  # 树的最大深度
    # 'min_samples_split': [2, 4, 6],  # 节点分裂所需的最小样本数
    # # 'min_samples_leaf': [10, 11, 12, 13, 14]
    # }
    # model = RandomForestRegressor()
    # grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=2)
    # # 模型拟合
    # grid_result = grid_search.fit(X_train, y_train)
    # print("得分: %f 最佳组合的参数值 %s" % (grid_result.best_score_, grid_search.best_params_))
    # # 得分: 0.567047 最佳组合的参数值 {'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 300}
    # return

    model = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_split=2)  # , min_samples_leaf=6
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_train)
    Residual(y_train, y_predict_test)
    return
    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 打印特征重要性
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, y_train[indices[f]], importances[indices[f]]))

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    # draw_learning_curve(model, X_train, y_train, scoring="r2")
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'RandomForest')


def model_ExtraTreesRegressor(X_train, y_train, X_test, y_test, fields):
    print('ExtraTreesRegressor模型')
    # print('搜索最佳组合参数...')
    # # 参数值字典
    # parameters = {
    #     'max_depth': [8, 9, 10, 11, 12],
    #     'min_samples_leaf': [10, 11, 12, 13, 14],
    #     'min_samples_split': [2, 3, 4, 6, 8, 10]
    # }
    # model = ExtraTreesRegressor()
    # grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='r2', cv=10, n_jobs=2)#, scoring='neg_mean_squared_error'
    # # 模型拟合
    # grid_result = grid_search.fit(X_train, y_train)
    # print("得分: %f 最佳组合的参数值 %s" % (grid_result.best_score_, grid_search.best_params_))
    # return
    # # 得分: 0.403548    最佳组合的参数值    {'max_depth': 12, 'min_samples_leaf': 10, 'min_samples_split': 10}

    model = ExtraTreesRegressor(max_depth=12, min_samples_leaf=10, min_samples_split=10)
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_test)

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'ExtraTrees')


# SGDRegressor
def model_AdaBoostRegressor(X_train, y_train, X_test, y_test, fields):
    print('AdaBoostRegressor模型')
    dt = DecisionTreeRegressor(random_state=42, splitter='best', max_depth=10, min_samples_leaf=10,
                               min_samples_split=2)  # ,max_features='log2'
    dt.fit(X_train, y_train)
    dt_err = 1.0 - dt.score(X_test, y_test)

    n_estimators = 10
    model = AdaBoostRegressor(estimator=dt, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_train)
    Residual(y_train, y_predict_test)
    return

    print(model.get_params())
    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'AdaBoost')

    # # 三个分类器的错误率可视化
    # fig = plt.figure()
    # # 设置 plt 正确显示中文
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # ax = fig.add_subplot(111)
    # ax.plot([1, n_estimators], [dt_err] * 2, 'k--', label=u'决策树模型 错误率')
    # ada_err = np.zeros((n_estimators,))
    # # 遍历每次迭代的结果 i 为迭代次数, pred_y 为预测结果
    # for i, pred_y in enumerate(model.staged_predict(X_test)):
    # # 统计错误率
    # ada_err[i] = zero_one_loss(pred_y, y_test)######有错误
    # # 绘制每次迭代的 AdaBoost 错误率
    # ax.plot(np.arange(n_estimators) + 1, ada_err, label='AdaBoost Test 错误率', color='orange')
    # ax.set_xlabel('迭代次数')
    # ax.set_ylabel('错误率')
    # leg = ax.legend(loc='upper right', fancybox=True)
    # plt.show()


# 梯度提升回归（Gradient boosting regression，GBR）
def model_GradientBoostingRegressor(X_train, y_train, X_test, y_test, fields):
    # 是一种从它的错误中进行学习的技术。它集成一堆较差的学习算法进行学习。
    # 每个学习算法准备率都不高，但是它们集成起来可以获得很好的准确率。
    # 这些学习算法依次应用，也就是说每个学习算法都是在前一个学习算法的错误中学习
    print('GradientBoostingRegressor模型')
    # print('搜索最佳组合参数...')
    # param_test1 = {'n_estimators': range(100, 1001, 200),
    #                'max_depth': range(3, 14, 2),
    #                'min_samples_split': range(100, 801, 200),
    #                'min_samples_leaf': range(60, 101, 10),
    #                'max_features': range(7, 20, 2),
    #                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    #                }
    # 得分: 0.636561 最佳组合的参数值 {'n_estimators': 900}
    param_test1 = {'n_estimators': range(100, 1001, 200)}
    model1 = GradientBoostingRegressor(min_samples_split=300, min_samples_leaf=20, max_depth=10,
                                       learning_rate=0.1, random_state=10)
    gsearch1 = GridSearchCV(estimator=model1, param_grid=param_test1, scoring='r2', n_jobs=2,
                            cv=10)  # , scoring='neg_mean_squared_error'
    gsearch1.fit(X_train, y_train)
    print("得分: %f 最佳组合的参数值 %s" % (gsearch1.best_score_, gsearch1.best_params_))
    means = gsearch1.cv_results_['mean_test_score']
    params = gsearch1.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))
    return

    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 1001, 200)}
    model2 = GradientBoostingRegressor(n_estimators=200, min_samples_leaf=20,
                                       learning_rate=0.1, random_state=10)
    gsearch2 = GridSearchCV(estimator=model2, param_grid=param_test2, scoring='r2', n_jobs=2,
                            cv=10)  # , scoring='neg_mean_squared_error'
    gsearch2.fit(X_train, y_train)
    print("得分: %f 最佳组合的参数值 %s" % (gsearch2.best_score_, gsearch2.best_params_))
    return

    param_test3 = {'min_samples_leaf': range(60, 101, 10)}
    model3 = GradientBoostingRegressor(n_estimators=100, max_depth=8,
                                       learning_rate=0.1, random_state=10)
    gsearch3 = GridSearchCV(estimator=model3, param_grid=param_test3, scoring='r2', n_jobs=2,
                            cv=10)  # , scoring='neg_mean_squared_error'
    gsearch3.fit(X_train, y_train)
    print("得分: %f 最佳组合的参数值 %s" % (gsearch3.best_score_, gsearch3.best_params_))
    return

    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_test)

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'GradientBoosting')


def model_BaggingRegressor(X_train, y_train, X_test, y_test, fields):
    print('BaggingRegressor模型')
    # print('搜索最佳组合参数...')
    # # 参数值字典
    # parameters = {
    #     'n_estimators': [15, 20, 25],  # 要集成的基估计器的个数,
    #     'max_samples': [0.8, 1.0],  # (default=1.0)。从x_train抽取去训练基估计器的样本数量。int代表抽取数量，float代表抽取比例,
    #     'max_features': [0.8, 1.0]  # (default=1.0)。从x_train抽取去训练基估计器的特征数量。int代表抽取数量，float代表抽取比例
    # }
    # model = BaggingRegressor()
    # grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=2)
    # # 模型拟合
    # grid_result = grid_search.fit(X_train, y_train)
    # print("得分: %f 最佳组合的参数值 %s" % (grid_result.best_score_, grid_search.best_params_))
    # return

    model = BaggingRegressor(n_estimators=20, max_samples=1.0, max_features=1.0)
    model.fit(X_train, y_train)
    y_predict_test = model.predict(X_test)

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, 'Bagging')


# 多层感知机（MultiLayer Perceptron，MLP）
def model_MLP(X_train, y_train, X_test, y_test, fields):
    # 是一种人工神经网络模型，通常用于处理分类问题。它是一种前馈神经网络（Feedforward Neural Network），由多个层次组成，每个层次包含多个神经元。
    # MLP 的基本组成包括：
    # 输入层（Input Layer）： 接收输入特征的层。每个输入特征都对应于输入层中的一个节点。
    # 隐藏层（Hidden Layers）： 在输入层和输出层之间的一层或多层。每个隐藏层包含多个神经元，每个神经元与前一层和后一层的所有神经元都有连接。
    # 输出层（Output Layer）： 生成最终输出的层。输出层的神经元数量通常取决于问题的类别数，例如，对于二分类问题，通常有一个输出神经元，表示两个类别的概率。
    # 每个神经元都与前一层的所有神经元相连接，并具有带权重的连接。在每个神经元中，输入被加权并通过激活函数进行转换，产生神经元的输出。这个过程可以表示为：
    # 输出=Activation(Weighted Sum of Inputs)
    # 其中，激活函数通常是非线性的，它引入了非线性变换，使得网络能够学习更加复杂的函数。
    # MLP 使用反向传播算法进行训练，通过最小化损失函数来调整连接权重，使得网络能够对训练数据进行更好的拟合。反向传播通过计算预测与实际标签之间的误差，并反向传播该误差以调整权重。
    # 由于 MLP 具有多个层次，它能够学习更加复杂的特征和关系，因此在许多应用中被广泛使用，包括图像识别、自然语言处理、分类等。
    # 实际上，深度学习任务通常使用更复杂的神经网络架构，可能包含多个隐藏层，不同的激活函数，以及其他调整参数。上述示例是一个简单的入门演示。
    print('多层感知机模型MLP')
    # # activation - 激活函数 {‘identity’, 'logistic', 'tanh', 'relu'}，默认relu
    # # identity - f(x) = x
    # # logistic - 其实就是sigmod函数，f(x) = 1 / (1 + exp(-x))
    # # tanh - f(x) = tanh(x)
    # # relu - f(x) = max(0, x)
    # parameters = {
    # 'hidden_layer_sizes': [(50, 50), (50, 100), (100, 50), (100, 100)],
    # 'activation': ['relu'],  # 'tanh', 'logistic', 该参数效果较差
    # 'solver': ['adam'],  # 'sgd'。  sgd-随机梯度下降；adam-机遇随机梯度的优化器
    # 'alpha': [ 0.01, 0.05, 0.1, 0.15],#, 0.2
    # 'learning_rate': ['adaptive']  # 'constant',
    # }
    # model = MLPRegressor(max_iter=500, random_state=100)
    # grid_result = GridSearchCV(estimator=model, param_grid=parameters, scoring='r2', cv=10, n_jobs=2)#, scoring='neg_mean_squared_error'
    # grid_result.fit(X_train, y_train.ravel())
    # # 查看最佳参数组合和性能
    # print("得分: %f 最佳组合的参数值 %s" % (grid_result.best_score_, grid_result.best_params_))
    # return

    # 得分: 0.359888 最佳组合的参数值 {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50), 'solver': 'adam'}
    # 得分: 0.352477 最佳组合的参数值 {'activation': 'relu', 'alpha': 0.15, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
    # 得分: 0.257293 最佳组合的参数值 {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (100, 100), 'learning_rate': 'adaptive', 'solver': 'adam'}
    # 创建 MLP 模型
    # model = MLPRegressor(hidden_layer_sizes=(50, 100), activation='relu', learning_rate='adaptive', solver='adam',
    #                     # 第一个隐藏层有50个节点，第二层有50个，激活函数用logistic，梯度下降方法用adam
    #                     alpha=0.15, max_iter=500, random_state=100) # best result
    model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', learning_rate='adaptive', solver='adam',
                         alpha=0.15, max_iter=500, random_state=100)

    # 训练模型
    model.fit(X_train, y_train.ravel())
    w = model.coefs_
    b = model.intercepts_  # 得到bias值
    y_predict_test = model.predict(X_train)
    Residual(y_train, y_predict_test)
    return
    
    # print('LarsCV输出解释变量、系数（保留5位小数）:', b, [np.around(i, 5) for i in w])
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercepts_))
    print([model.intercepts_] + model.coefs_)

    # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, "MLP")


# LinearSVR虽然很快，但是在非线性数据集中拟合的误差很大，应避免在非线性回归问题上使用线性SVR；
# 高斯核SVR拟合时间较长，但均方误差较低，拟合效果好；
# 多项式核SVR的拟合时间比高斯SVR快，但误差更高。
def model_SVR(X_train, y_train, X_test, y_test, fields):
    print('SVR模型')
    start = time()

    # selectmode = 1
    # ######################### 高斯核函数 ########################
    # param_grid = {
    # 'C': [0.1, 0.5, 1, 2, 3],  # 惩罚参数'C': [0.1, 0.5, 1, 2],
    # 'gamma': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],  # 核函数参数'gamma': [0.09, 0.009, 0.0009, 0.00009]
    # 'epsilon': [0.2, 0.4, 0.6] # 阈值
    # }
    # grid1 = []
    # # scores = ['accuracy', 'average_precision', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
    # #           'neg_log_loss', 'precision', 'recall', 'roc_auc']
    # scores = 'r2'  # 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error',r2
    # print("寻找高斯核函数最优参数依据 %s" % scores)
    # grid1 = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, scoring=scores, cv=5, n_jobs=2) # , scoring='neg_mean_squared_error'
    # grid1.fit(X_train, y_train)

    # print("参数间不同数值的组合后得到的分数:")
    # means = grid1.cv_results_['mean_test_score']
    # stds = grid1.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, grid1.cv_results_['params']):
    # print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # print('GridSearchCV网格搜索，最优的松弛变量C：{}，gamma：{}，score：{}'.format(
    # grid1.best_estimator_.get_params()['C'], grid1.best_estimator_.get_params()['gamma'], grid1.best_score_))
    # print("最好的参数搭配结果:")
    # print(grid1.best_params_)
    # exit(0)

    # best_C = grid1.best_estimator_.get_params()['C']
    # best_gamma = grid1.best_estimator_.get_params()['gamma']
    # best_epsilon = grid1.best_estimator_.get_params()['epsilon']
    # best_score = grid1.best_score_

    ######################## 多项式核函数 拟合度低 ########################
    # print('多项式核函数,寻找最优参数')
    # param_grid = {
    # 'C': [0.1, 0.5, 1, 2],  # 惩罚参数'C': [0.1, 0.5, 1, 2],
    # 'epsilon': [0.1, 0.2, 0.5, 1], # 阈值
    # 'gamma': [0.15, 0.1, 0.05],
    # 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 核函数参数'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # }
    # grid2 = GridSearchCV(SVR(kernel='poly'), param_grid=param_grid, scoring=score, cv=5, n_jobs=2)
    # grid2.fit(X_train, y_train)
    # print('GridSearchCV网格搜索，最优的松弛变量C：{}，最优的阶数：{}，模型score：{}'.format(
    # grid2.best_estimator_.get_params()['C'], grid2.best_estimator_.get_params()['degree'], grid2.best_score_))

    # best_C = grid2.best_estimator_.get_params()['C']
    # best_degree = grid2.best_estimator_.get_params()['degree']
    # if best_score < grid2.best_score_:
    # best_score = grid2.best_score_
    # selectmode = 2

    ##################################################################
    # plt.scatter(x=grid2.cv_results_['param_C'].data, y=grid2.cv_results_['param_degree'].data,
    #             s=1000 * (grid2.cv_results_['mean_test_score'] - min(grid2.cv_results_['mean_test_score']) + 0.1),
    #             cmap=plt.cm.get_cmap('RdYlBu'))
    # plt.xlabel("C")
    # plt.ylabel("degree")
    # plt.title("score随惩罚系数C和degree变化散点图")
    # plt.annotate('圆圈大小表示score大小', xy=(1, 1), xytext=(1, -1), color='b', size=10)
    # plt.annotate('这个点最优', xy=(1, 1),
    #              xytext=(grid2.best_estimator_.get_params()['C'] + 1, grid2.best_estimator_.get_params()['degree']),
    #              color='r')
    ##################################################################
    # if selectmode == 1:
    #     model = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
    # else:
    #     model = SVR(kernel='poly', C=best_C, degree=best_degree)

    # model = SVR(kernel='poly', C=best_C, degree=best_degree)
    # model = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_epsilon)

    # {'C': 1, 'gamma': 0.1}{'C': 2, 'gamma': 0.1, 'epsilon': 0.5}
    model = SVR(kernel='rbf', C=2, gamma=0.1, epsilon=0.5)
    model.fit(X_train, y_train)
    # 预测
    y_predict_test = model.predict(X_train)
    Residual(y_train, y_predict_test)
    return
    
    w = model.coef0  # 得到权重列表
    b = model.intercept_  # 得到bias值
    # print('SVR输出解释变量、系数（保留5位小数）:', b)
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    # print([model.intercept_] + model.coef0)

    end = time()
    print('计算耗时%f秒' % (end - start))

    # 预测效果验证
    print("score:{}".format(np.around(model.score(X_train, y_train), 5)))  # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    # draw_learning_curve(model, X_train, y_train)
    # draw_validation_curve(model, X_train, y_train, param_range=np.linspace(0, 0.4, 10))
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, "SVR")


def model_LinearSVR(X_train, y_train, X_test, y_test, fields):
    print('LinearSVR模型')
    start = time()

    # selectmode = 1
    # ######################## 高斯核函数 ########################
    param_grid = {
        'C': [0.1, 0.5, 1, 2, 10, 20],  # 核函数参数'C'
        'epsilon': [0.5, 1, 1.5, 2]  # 正则化参数'epsilon'
    }
    grid1 = []
    # scores = ['accuracy', 'average_precision', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
    #           'neg_log_loss', 'precision', 'recall', 'roc_auc']
    scores = ['r2']  # 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error',r2
    for score in scores:
        print("寻找最优参数依据 %s" % score)
        grid1 = GridSearchCV(LinearSVR(), param_grid=param_grid, return_train_score=True, scoring='r2', n_jobs=2,
                             cv=10)  # , scoring='neg_mean_squared_error'
        grid1.fit(X_train, y_train)

        print("最好的参数搭配结果:")
        print(grid1.best_params_)

        print("参数间不同数值的组合后得到的分数:")
        means = grid1.cv_results_['mean_test_score']
        stds = grid1.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid1.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # grid1 = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, scoring=ftwo_scorer, cv=5, n_jobs=2)
    # grid1.fit(X_train, y_train)
    print('GridSearchCV网格搜索，最优的松弛变量C：{}，epsilon：{}，score：{}'.format(
        grid1.best_estimator_.get_params()['C'], grid1.best_estimator_.get_params()['epsilon'], grid1.best_score_))
    best_C = grid1.best_estimator_.get_params()['C']
    best_epsilon = grid1.best_estimator_.get_params()['epsilon']
    best_score = grid1.best_score_

    # ######################## 多项式核函数 拟合度低 ########################
    # print('多项式核函数,寻找最优参数')
    # param_grid = {
    #     'C': [0.1, 0.5, 1, 2],  # 正则化参数'C': [0.1, 0.5, 1, 2],
    #     'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 核函数参数'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # }
    # grid2 = GridSearchCV(SVR(kernel='poly'), param_grid=param_grid, scoring=ftwo_scorer, cv=10, n_jobs=2)
    # grid2.fit(X_train, y_train)
    # print('GridSearchCV网格搜索，最优的松弛变量C：{}，最优的阶数：{}，模型score：{}'.format(
    #     grid2.best_estimator_.get_params()['C'], grid2.best_estimator_.get_params()['degree'], grid2.best_score_))
    # best_C = grid2.best_estimator_.get_params()['C']
    # best_degree = grid2.best_estimator_.get_params()['degree']
    # if best_score < grid2.best_score_:
    #     best_score = grid2.best_score_
    #     selectmode = 2

    # ##############
    # plt.scatter(x=grid2.cv_results_['param_C'].data, y=grid2.cv_results_['param_degree'].data,
    #             s=1000 * (grid2.cv_results_['mean_test_score'] - min(grid2.cv_results_['mean_test_score']) + 0.1),
    #             cmap=plt.cm.get_cmap('RdYlBu'))
    # plt.xlabel("C")
    # plt.ylabel("degree")
    # plt.title("score随惩罚系数C和degree变化散点图")
    # plt.annotate('圆圈大小表示score大小', xy=(1, 1), xytext=(1, -1), color='b', size=10)
    # plt.annotate('这个点最优', xy=(1, 1),
    #              xytext=(grid2.best_estimator_.get_params()['C'] + 1, grid2.best_estimator_.get_params()['degree']),
    #              color='r')

    # if selectmode == 1:
    #     model = SVR(kernel='rbf', C=best_C, gamma=best_epsilon)
    # else:
    #     model = SVR(kernel='poly', C=best_C, degree=best_degree)

    # draw_validation_curve(SVR(kernel='rbf'), X_train, y_train)##'gamma': 0.005
    # model = LinearSVR(kernel='rbf', C=1, gamma=0.005)  #{'C': 1, 'gamma': 0.1}
    model = LinearSVR(C=best_C, epsilon=best_epsilon)
    model.fit(X_train, y_train)

    # 预测
    y_predict_test = model.predict(X_test)
    w = model.coef_  # 得到权重列表
    b = model.intercept_  # 得到bias值
    # print('SVR输出解释变量、系数（保留5位小数）:', b)
    print('解释变量%-50s\n系数（保留5位小数）' % (model.intercept_))
    # print([model.intercept_] + model.coef0)

    end = time()
    print('计算耗时%f秒' % (end - start))

    # 预测效果验证
    print("score:{}".format(np.around(model.score(X_train, y_train), 5)))  # 预测效果验证
    print("Train R Squared:{}".format(np.around(model.score(X_train, y_train), 5)))
    # regressor_importance(model, X_train, y_train)
    regressor_Result(y_test, y_predict_test, fields)
    draw_Result(y_test, y_predict_test, "SVR")
    draw_learning_curve(model, X_train, y_train)


#######################################################################
def regressor_Result(y_test, y_predict_test, fields=None):
    # R²越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好
    # R²越接近0，表明模型拟合的越差
    # 经验值：R2>0.4， 拟合效果好
    # 缺点：数据集的样本越大，R²越大，因此，不同数据集的模型结果比较会有一定的误差
    print("R² Squared: ", round(metrics.r2_score(y_test, y_predict_test), 4))

    # 解释方差分，用来衡量模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
    print("Explained Variance Score(EVS): ", round(metrics.explained_variance_score(y_test, y_predict_test), 4))
    # MAE（平均绝对误差）, 一般来说取值越小，模型的拟合效果就越好。
    print('Mean Absolute Error(MAE): ', round(metrics.mean_absolute_error(y_test, y_predict_test), 4))
    # MSE（均方误差）, 预测值与真实值之间的差异，越小表示模型的预测越准确
    print('Mean Squared Error(MSE): ', round(metrics.mean_squared_error(y_test, y_predict_test), 4))
    # RMSE（均方根误差）,取值范围是0到正无穷大，数值越小表示模型的预测误差越小，模型的预测能力越强，预测效果越好。
    # 注意，RMSE对较大的偏差非常敏感，当数据中存在较大的异常值时，RMSE可能会受到较大的影响。
    # 计算拟合优度的另一种方法
    print('Root Mean Squared Error(RMSE): ', round(metrics.root_mean_squared_error(y_test, y_predict_test), 4))

    Regression = np.sum((y_predict_test - np.mean(y_test)) ** 2)  # ESS 回归平方和 模型平方和
    total = np.sum((y_test - np.mean(y_test)) ** 2)  # 总体平方和
    Residual = np.sum((y_test - y_predict_test) ** 2)  # 残差平方和
    R_square = 1 - Residual / total
    print('Explained sum of squares (ESS):', Regression)
    print('Total sum of squares (TSS):', total)
    print('Residual sum of squares (RSS):', Residual)
    print('R²_square:', round(R_square, 4))

    if fields != None:
        # 计算参数数量（包括截距）
        k = len(fields) + 1
        n = y_predict_test.shape[0]
        print(f"k:{k}  n:{n}")
        # 计算对数似然值
        sigma_squared = Residual / n
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_squared) + 1)
        # 计算AIC
        AIC = 2 * k - 2 * log_likelihood
        # 计算AICc
        AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
        # 计算 BIC
        BIC = k * np.log(n) - 2 * log_likelihood

        print(f"AIC: {AIC}")
        print(f"AICc: {AICc}")
        print(f"BIC: {BIC}")


def regressor_importance(model, X_train, y_train, fields=None):
    print("特征重要性")
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_train, y_train, n_repeats=30, random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            if fields == None:
                print(f"X{i:<8}"
                      f"{r.importances_mean[i]:.3f}"
                      f" +/- {r.importances_std[i]:.3f}")
            else:
                print(f"{fields[i]:<8}"
                      f"{r.importances_mean[i]:.3f}"
                      f" +/- {r.importances_std[i]:.3f}")


def regressor_cross_val(model, X_train, y_train):
    from sklearn.model_selection import cross_val_score
    # 默认是3-fold cross validation
    # print("交叉验证，评价模型的效果", round(
    # np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5), 4)))  # 输入训练集的数据和目标值

    # 将得分*（-1），因为scikit learn计算的是负的MAE
    scores = -1 * cross_val_score(model, X_train, y_train, cv=100, scoring='neg_mean_absolute_error')
    print("MAE scores:", scores)

    # k_range = range(1, 30)
    # cv_scores = []  # 用来放每个模型的结果值
    # for n in k_range:
    # knn = KNeighborsClassifier(n)  # knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
    # scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')  # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值。
    # cv_scores.append(scores.mean())
    # plt.plot(k_range, cv_scores)
    # plt.xlabel('K')
    # plt.ylabel('Accuracy')  # 通过图像选择最好的参数
    # plt.show()


# 必须是整数分类结果标签
def classification_Result(y_train, y_test, y_predict_test):
    # "max_error",
    # "mean_squared_log_error",
    # "median_absolute_error",
    # "mean_absolute_percentage_error",
    # "mean_pinball_loss",
    # "r2_score",
    # "root_mean_squared_log_error",
    # "mean_tweedie_deviance",
    # "mean_poisson_deviance",
    # "mean_gamma_deviance",
    # "d2_tweedie_score",
    # "d2_pinball_score",
    # "d2_absolute_error_score",

    # continuous is not supported
    print('混淆矩阵: n', metrics.confusion_matrix(y_test, y_predict_test))
    print('accuracy score准确率: ', metrics.accuracy_score(y_test, y_predict_test))  # 混淆矩阵对角线元素之和/所有元素之和
    print('recall score: ', metrics.recall_score(y_test, y_predict_test))
    print('precision score: ', metrics.precision_score(y_test, y_predict_test))
    print('F1 score: ', metrics.f1_score(y_test, y_predict_test))
    print('precision recall fscore support: ', metrics.precision_recall_fscore_support(y_test, y_predict_test))
    print("AUC Score (Train): ", metrics.roc_auc_score(y_train, [metrics.r2_score(y_test, y_predict_test)]))
    # target_names = ['class 0', 'class 1', 'class 2']
    # classifyreport = metrics.classification_report(y_test, y_predict_test, target_names=target_names)
    # print('分类结果报告: \n', classifyreport)


def draw_Result(y_test, y_predict_test, title):
    # y_test=y_test+1
    # y_predict_test=y_predict_test+1

    # 创建线性回归模型并拟合数据
    model = LinearRegression()
    y_predict_test = y_predict_test.reshape(-1, 1)
    model.fit(y_predict_test, y_test)
    # 生成预测值，用于绘制拟合直线
    y_pred_line = model.predict(y_predict_test)

    # 绘制散点图
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(y_predict_test, y_test, s=5, color='blue', label='Predict vs  Actual')

    # # x_min = int(y_predict_test.min() - 0.8)  # 留出一些边缘空间
    # # x_max = int(y_pred_line.max() + 0.5)  # 留出一些边缘空间
    # # y_min = int(y_test.min() - 0.8)
    # # y_max = int(y_test.max() + 0.5)

    # 矩形边界值
    x_min, x_max, y_min, y_max = plt.axis()
    # plt.plot([x_min, x_max], [y_min, y_max], color='black', linestyle="--", linewidth=1.5)  # 对角线    
    plt.plot(y_predict_test, y_pred_line, color='red', linewidth=1.5, label='Fitting line')  # 绘制拟合直线

    # 计算R²值
    r2 = round(metrics.r2_score(y_test, y_predict_test), 4)
    # 标注R²值
    plt.text(x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.2, f'R² = {r2:.2f}', fontsize=10,
             color='black')

    # 计算RMSE值
    RMSE = round(metrics.root_mean_squared_error(y_test, y_predict_test), 4)
    # 标注RMSE值
    plt.text(x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.25, f'RMSE = {RMSE:.2f}', fontsize=10,
             color='black')

    # 标注拟合方程
    w = model.coef_  # 得到权重列表
    b = model.intercept_  # 得到bias值
    s = "Y=" + str(f'{b[0]:.4f}') + "+" + str(f'{w[0][0]:.4f}') + "X"
    plt.text(x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.3, f'{s}', fontsize=10,
             color='black')

    # 设置图表标题和标签
    plt.axis([x_min, x_max, y_min, y_max])  ##（-0.2，2）x轴的范围预测值， （0,2）y轴的范围真实值
    plt.title(title)
    plt.xlabel('Predict Value')
    plt.ylabel('Actual Value')
    plt.legend()
    plt.show()


def draw_QQResult(y_test, y_predict_test):
    y_predict_test.sort(axis=0)
    y_test.sort(axis=0)
    x1 = range(len(y_predict_test))
    x2 = range(len(y_test))

    # 绘制散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(x1, y_predict_test, alpha=0.5, s=5, c="red", label="Predict")
    plt.scatter(x2, y_test, alpha=0.5, s=5, c='green', label="Actual")
    # plt.plot([min(x1), max(x1)], [y_predict_test.min(), y_predict_test.max()], 'red')
    # 解包边界值
    x_min, x_max, y_min, y_max = plt.axis()
    plt.plot([x_min, x_max], [y_min, y_max], 'k--')  # 对角线

    # 设置图表标题和标签
    plt.axis([x_min, x_max, y_min, y_max])  ##（-0.2，2）x轴的范围预测值， （0,2）y轴的范围真实值
    plt.title('Scatter of predict and actual data')
    plt.xlabel('Predict Value')
    plt.ylabel('Actual Value')
    plt.legend()
    plt.show()


# 绘制响应面法(Response Surface Methodology, RSM )
def draw_RSM(X, y, fields=None, isScatter=False, isLine=True, degree=2):
    print("绘制响应面(Response Surface Methodology, RSM )")
    data = np.column_stack((X, y))  # 将X和y合并为一个数组
    fig, axs = plt.subplots(5, 3, figsize=(8, 8))
    for i in range(5):
        for j in range(3):
            idx = i * 3 + j
            if idx >= X.shape[1]:
                axs[i, j].axis('off')  # 如果没有更多的特征，隐藏子图
                continue

            sorted_data = data[np.argsort(data[:, idx]), :]  # 指定根据第idx列进行排序
            # sorted_data = data[np.argsort(data[:, data.shape[1]-1]), :]# 指定根据y列进行排序

            # 提取排序后的X和y
            sorted_X = sorted_data[:, idx]
            # sorted_y = sorted_data[:, data.shape[1]-1]
            sorted_y = sorted_data[:, -1]

            # # axs[i, j].plot(sorted_X, sorted_y, color="r", label=f'{fields[idx]}')

            # 绘制拟合曲线            
            X_fit = np.linspace(sorted_X.min(), sorted_X.max(), 100)
            coeffs = np.polyfit(sorted_X, sorted_y, degree)  # 多项式拟合
            # poly = np.poly1d(coeffs)
            # y_fit = poly(X_fit)
            y_fit = np.polyval(coeffs, X_fit)

            if fields == None:
                axs[i, j].set_title(f'X{idx}')

                if isScatter == True:
                    axs[i, j].scatter(sorted_X, sorted_y, s=1, color='r', label=f'Predict vs X{idx}')
                if isLine == True:
                    axs[i, j].plot(X_fit, y_fit, color="b", linewidth=1, label=f'Polynomial fit X{idx}')
            else:
                axs[i, j].set_title(f'{fields[idx]}', fontsize=10)

                if isScatter == True:
                    axs[i, j].scatter(sorted_X, sorted_y, s=1, color='r', label=f'Predict vs {fields[idx]}')
                if isLine == True:
                    axs[i, j].plot(X_fit, y_fit, color="b", linewidth=1, label=f'Polynomial fit {fields[idx]}')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.show()


# 绘制响应面法(Response Surface Methodology, RSM )--单独成图
def draw_RSM_individual(X, y, fields=None, isScatter=False, isLine=True, degree=2):
    print("绘制响应面(Response Surface Methodology, RSM )")
    data = np.column_stack((X, y))  # 将X和y合并为一个数组

    # 遍历每个特征
    for idx in range(X.shape[1]):
        # 创建新的图形
        fig, ax = plt.subplots(figsize=(6, 4))

        # 根据指定特征排序数据
        sorted_data = data[np.argsort(data[:, idx]), :]
        sorted_X = sorted_data[:, idx]
        sorted_y = sorted_data[:, -1]

        # 添加标题和标签
        if fields is None:
            ax.set_title(f'X{idx}')
        else:
            ax.set_title(f'{fields[idx]}', fontsize=24)

        # 绘制散点图
        if isScatter:
            ax.scatter(sorted_X, sorted_y, s=1, color='r',
                       label=f'Predict vs X{idx}' if fields is None else f'Predict vs {fields[idx]}')

        # 绘制拟合曲线
        if isLine:
            # 绘制拟合曲线
            X_fit = np.linspace(sorted_X.min(), sorted_X.max(), 100)
            coeffs = np.polyfit(sorted_X, sorted_y, degree)
            y_fit = np.polyval(coeffs, X_fit)

            ax.plot(X_fit, y_fit, color="b", linewidth=1,
                    label=f'Polynomial fit X{idx}' if fields is None else f'Polynomial fit {fields[idx]}')

        # # 添加图例
        # ax.legend(fontsize=8)

        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', labelsize=20)

        # 调整布局
        plt.tight_layout()

        # 保存或显示图形
        plt.savefig(f'RSM_feature_{idx}.png')  # 保存为PNG文件
        plt.show()


# 模型验证曲线,验证SVC中的一个参数gamma 在什么范围内能使model产生好的结果. 以及过拟合和gamma取值的关系，越低越好
def draw_validation_curve(estimator, X, y, param_name='gamma', param_range=np.logspace(-6, -2.3, 5),
                          scoring='neg_mean_squared_error', cv=10, n_jobs=2):
    print("模型验证曲线(validation_curve)")

    # 使用validation_curve快速找出参数对模型的影响
    train_loss, test_loss = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range,
                                             cv=cv, n_jobs=n_jobs, scoring=scoring)

    # 平均每一轮的平均方差
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    # 可视化图形
    plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross-validation")
    plt.title('validation curves')
    plt.xlabel(param_name)
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


############# 学习曲线(learning curve)来判断模型状态：过拟合欠拟合。当训练集和验证集得分接近且低时，模型可能欠拟合；当训练集得分远高于验证集时，模型可能过拟合。
def draw_learning_curve(estimator, X, y, train_sizes=np.linspace(.1, 1.0, 10),
                        scoring="neg_mean_squared_error", cv=10, n_jobs=2):
    print("学习曲线(learning curve)")
    # neg_mean_absolute_error, neg_mean_squared_error, neg_median_absolute_error, r2
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, scoring=scoring,
                                                            cv=cv, n_jobs=n_jobs)
    train_scores_mean = abs(np.mean(train_scores, axis=1))  # 取绝对值
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = abs(np.mean(test_scores, axis=1))  # 取绝对值
    test_scores_std = np.std(test_scores, axis=1)

    # plt.figure()
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.title('Learning curves')
    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    # plt.ylabel(scoring)
    plt.legend(loc="best")
    plt.show()


def draw_histogram(X, fields=None):
    from scipy.stats import norm
    print("回归系数直方图")
    fig, axs = plt.subplots(5, 3, figsize=(8, 8))
    for i in range(5):
        for j in range(3):
            idx = i * 3 + j
            if idx >= X.shape[1]:
                axs[i, j].axis('off')  # 如果没有更多的特征，隐藏子图
                continue

            sorted_data = X[np.argsort(X[:, idx]), :]  # 指定根据第idx列进行排序
            x = sorted_data[:, idx]
            # Fit a normal distribution to the data:
            # mean and standard deviation
            mu, std = norm.fit(x)

            # Plot the histogram.
            if fields == None:
                axs[i, j].hist(x, bins=15, density=True, edgecolor='w', alpha=0.7, color='b')
                axs[i, j].set_title(f'X{idx}')
            else:
                # 绘制拟合曲线
                axs[i, j].hist(x, bins=15, density=True, edgecolor='w', alpha=0.7, color='b')
                axs[i, j].set_title(f'{fields[idx]}', fontsize=10)

                # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    # 自动调整子图布局
    plt.tight_layout()
    plt.show()


# 混淆矩阵的可视化
def draw_confusion_matrix(y_test, y_predict_test):
    # 混淆矩阵
    cm = metrics.confusion_matrix(y_test, y_predict_test, labels=[0, 1])
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='.2e', cmap='GnBu')
    # 图形显示
    plt.show()


def draw_auc(model, X_test, y_test):
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)  # 计算AUC的值

    # 绘制面积图
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    # 添加边际线
    plt.plot(fpr, tpr, color='black', lw=1)
    # 添加对角线
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # 添加文本信息
    plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)
    # 添加x轴与y轴标签
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    # 显示图形
    plt.show()

    # 调用自定义函数，绘制K-S曲线
    plot_ks(y_test=y_test, y_score=y_score, positive_flag=1)


def plot_ks(y_test, y_score, positive_flag):
    # 对y_test,y_score重新设置索引
    y_test.index = np.arange(len(y_test))
    # y_score.index = np.arange(len(y_score))
    # 构建目标数据集
    target_data = pd.DataFrame({'y_test': y_test, 'y_score': y_score})
    # 按y_score降序排列
    target_data.sort_values(by='y_score', ascending=False, inplace=True)
    # 自定义分位点
    cuts = np.arange(0.1, 1, 0.1)
    # 计算各分位点对应的Score值
    index = len(y_score) * cuts
    scores = y_score.iloc[index.astype('int')]
    # 根据不同的Score值，计算Sensitivity和Specificity
    Sensitivity = []
    Specificity = []
    for score in scores:
        # 正例覆盖样本数量与实际正例样本量
        positive_recall = \
            target_data.loc[(target_data.y_test == positive_flag) & (target_data.y_score > score), :].shape[0]
        positive = sum(target_data.y_test == positive_flag)
        # 负例覆盖样本数量与实际负例样本量
        negative_recall = \
            target_data.loc[(target_data.y_test != positive_flag) & (target_data.y_score <= score), :].shape[0]
        negative = sum(target_data.y_test != positive_flag)
        Sensitivity.append(positive_recall / positive)
        Specificity.append(negative_recall / negative)
    # 构建绘图数据
    plot_data = pd.DataFrame({'cuts': cuts, 'y1': 1 - np.array(Specificity), 'y2': np.array(Sensitivity),
                              'ks': np.array(Sensitivity) - (1 - np.array(Specificity))})
    # 寻找Sensitivity和1-Specificity之差的最大值索引
    max_ks_index = np.argmax(plot_data.ks)
    plt.plot([0] + cuts.tolist() + [1], [0] + plot_data.y1.tolist() + [1], label='1-Specificity')
    plt.plot([0] + cuts.tolist() + [1], [0] + plot_data.y2.tolist() + [1], label='Sensitivity')
    # 添加参考线
    plt.vlines(plot_data.cuts[max_ks_index], ymin=plot_data.y1[max_ks_index],
               ymax=plot_data.y2[max_ks_index], linestyles='--')
    # 添加文本信息
    plt.text(x=plot_data.cuts[max_ks_index] + 0.01,
             y=plot_data.y1[max_ks_index] + plot_data.ks[max_ks_index] / 2,
             s='KS= %.2f' % plot_data.ks[max_ks_index])
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()


# 正态性检验
def draw_Data_test(pd, fieldData, fields):
    # # 散点图矩阵
    # # 绘制散点图矩阵
    # # sns.pairplot(pd.loc[:, ['RD_Spend', 'Administration', 'Marketing_Spend', 'Profit']])
    # sns.pairplot(pd.loc[:, fields])
    # plt.show()

    # Shapiro-Wilk test
    S, p = stats.shapiro(fieldData)
    print('the shapiro test result is:', S, ',', p)

    # kstest（K-S检验）
    K, p = stats.kstest(fieldData, 'norm')
    print(K, p)

    # normaltest
    N, p = stats.normaltest(fieldData)
    print(N, p)

    # Anderson-Darling test
    A, C, p = stats.anderson(fieldData, dist='norm')
    print(A, C, p)


# 直方图法----正态性检验
def draw_Data_histogram(data, field):
    fieldData = data[field].values
    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制直方图
    sns.distplot(a=fieldData, bins=10, fit=stats.norm, norm_hist=True,
                 hist_kws={'color': 'steelblue', 'edgecolor': 'black'},
                 kde_kws={'color': 'black', 'linestyle': '--', 'label': '核密度曲线'},
                 fit_kws={'color': 'red', 'linestyle': ':', 'label': '正态密度曲线'})
    plt.legend()
    plt.show()


# 通过sklearn内建函数筛选指定的k个有利特征，这里选择k = 5
def selectKBest_feature(X, y):
    from sklearn.feature_selection import SelectKBest, f_regression

    # 筛选和标签最相关的k=5个特征
    selector = SelectKBest(f_regression, k=5)
    X_new = selector.fit_transform(X, y)
    print(X_new.shape)
    print('最相关的几列', selector.get_support(indices=True).tolist())
    return X_new


# 相关性分析   如果相关系数R>0.8时就可能存在较强相关性
def model_correlations(data):
    correlations = data.corr(method='pearson')  # 皮尔森相关性
    print('相关性分析系数：\n', correlations)
    # names = correlations.columns.tolist()
    # fig = plt.figure(figsize=(30, 20))
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(correlations, vmin=-1, vmax=1)
    # fig.colorbar(cax)
    # ticks = np.arange(0, len(names), 1)
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(names)
    # ax.set_yticklabels(names)
    # plt.show()

    plt.figure(figsize=(30, 20))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    sns.heatmap(correlations,
                annot=True,  # 显示相关系数的数据
                center=0.5,  # 居中
                fmt='.2f',  # 只显示两位小数
                linewidth=0.5,  # 设置每个单元格的距离
                linecolor='blue',  # 设置间距线的颜色
                vmin=0, vmax=1,  # 设置数值最小值和最大值
                xticklabels=True, yticklabels=True,  # 显示x轴和y轴
                square=True,  # 每个方格都是正方形
                cbar=True,  # 绘制颜色条
                cmap='coolwarm_r',  # 设置热力图颜色
                )
    plt.title('correlation thermogram')
    plt.show()


# 主成分分析
def model_PCA(data):
    from sklearn.decomposition import PCA  # 主成分分析
    # pca = PCA(n_components=3)
    pca = PCA()  # .fit(data)
    # pca.fit(data)
    principalComponents = pca.fit_transform(data)
    # print('降维后的数据:', principalComponents)
    # principalDf = pd.DataFrame(data=principalComponents, columns=['p_component_1', 'p_component_2', 'p_component_3'])
    # finalDf = pd.concat([principalDf], axis=1)
    print('最大方差的成分:', pca.components_)
    print('主成分的方差累计贡献率{}'.format(np.cumsum(pca.explained_variance_ratio_)))
    print('协方差数据:', pca.get_covariance())
    print('数据精度矩阵（用生成模型）:', pca.get_precision())
    print('所有样本的log似然平均值:', pca.score(data))
    print('奇异值:', pca.singular_values_)
    print('噪声协方差:', pca.noise_variance_)


# 特征值（Eigenvalue）  实际上就是对自变量做主成分分析，如果多个维度的特征值等于0，则可能有比较严重的共线性。
def eigenvalue(x, y):
    sc = StandardScaler()
    X_std = sc.fit_transform(x)
    cov_mat = np.cov(X_std.T)
    # 计算特征必须为方阵才行，也可以对缺失的地方补0
    eigenvalue, featurevector = np.linalg.eig(cov_mat)
    # print(featurevector)
    print('特征值:\n', eigenvalue)

    # 计算累计解释方差和
    tot = sum(eigenvalue)
    var_exp = [(i / tot) for i in sorted(eigenvalue, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(1, 11), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1, 11), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('./figures/pca1.png', dpi=300)
    plt.show()

    # 对特征值排序
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigenvalue[i]), featurevector[:, i])
                   for i in range(len(eigenvalue))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    # 选择最大的两个特征值对应的特征向量，只用两个特征向量是为了下面画图方便，实际运用PCA时，到底选择几个特征向量，要考虑计算效率和分类器的表现两个方面(常用的选择方式是特征值子集要包含90%方差)：
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    print('特征值对应的特征向量, Matrix W:\n', w)

    # 特征维度降到2维度后，就可以用散点图将数据可视化出来了：
    X_std_pca = X_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X_std_pca[y == l, 0],
                    X_std_pca[y == l, 1],
                    c=c, label=l, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.savefig('./figures/pca2.png', dpi=300)
    plt.show()

    return eigenvalue, featurevector


# 共线性检验
def inflation_factor_csv(csv_file, fields):
    data = pd.read_csv(csv_file, low_memory=False)
    return inflation_factor(data, fields)


def inflation_factor(data, fields):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # 共线性数据处理
    # 1)增大样本量：增大样本量可以消除由数据量不足而出现的偶然的共线性现象。
    # 2)做差分：对于时间序列来讲一阶差分可以有效地消除多重共线性。
    # 3)岭回归法（Ridge Regression）：通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价来获得更实际和可靠性更强的回归系数。
    # 4)逐步回归法（Stepwise Regression）:每次引入一个自变量进行统计检验，然后逐步引入其他变量，同时对所有变量的回归系数进行检验，如果原来引入的变量由于后面变量的引入而变得不再显著，那么久将其剔除，逐步得到最有回归方程。
    # 5)主成分回归（Principal Components Regression）: 通过主成分分析，使用PCA降维后再建模。
    # 对于高共线性且价值不大的数据直接删除即可。

    # 1.方差膨胀系数(variance inflation factor，VIF)
    # VIF是容忍度的倒数，值越大则共线性问题越明显，通常以10作为判断边界。当VIF<10,不存在多重共线性；
    # 当10<=VIF<100,存在较强的多重共线性；当VIF>=100, 存在严重多重共线性。

    # creating dummies for gender
    # data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})
    # the independent variables set
    X = data[fields]

    # 相关系数
    # corr_R(X.values)
    # eigenvalue(X.values, y)#要去除第一行

    y = data['probability'].values  # .reshape((-1, 1))
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    # calculating TOL for each feature
    vif_data["TOL"] = [1. / variance_inflation_factor(X.values, X.columns.get_loc(i)) for i in X.columns]
    # 容忍度是每个自变量作为因变量对其他自变量进行回归建模时得到的残差比例，大小用1减得到的决定系数来表示。
    # 容忍度值越小说明这个自变量与其他自变量间越可能存在共线性问题。
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    # 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
    vif_data = vif_data.sort_values('VIF', ascending=False)
    print('多重共线性\n', vif_data)

    return list(vif_data["feature"])


# 变异系数
def variation_index(data):
    # cv = lambda x: np. std (data, ddof= 1 ) / np. mean (x) * 100
    cv = lambda x: np.std(x) / np.mean(x)

    var1 = np.apply_along_axis(cv, axis=0, arr=data)
    print("\nVariation at axis = 0: \n", var1)

    var2 = np.apply_along_axis(cv, axis=1, arr=data)
    print("\nVariation at axis = 1: \n", var2)
    ########################################################

    Avg = np.average(data, axis=0)  # 计算均值
    Stad = np.std(data, axis=0)  # 计算标准差
    V = Stad / Avg  # 计算变异系数
    for i in range(len(V)):
        print(f"第{i + 1}个对象变异系数：{V[i]}")
    # print("变异系数为：\n{}".format(V))

    # 计算权重
    w = V / sum(V)
    # for i in range(len(w)):
    # print(f"第{i+1}个对象权重：{w[i]}")
    # print('权重为：\n{}'.format(w))

    # 得分
    s = np.dot(data, w)
    Score = 100 * s / max(s)
    for i in range(len(Score)):
        print(f"第{i + 1}个对象的百分制得分：{Score[i]}")

    return V


# 线性转逻辑斯蒂
def sigmoid(x):
    if x > 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.exp(x) / (1.0 + np.exp(x))


# Hurst 指数
def Hurst(ts):
    '''
    Parameters
    ----------
    ts : Iterable Object.
        A time series or a list.

    Raises
    ------
    ValueError
        If input ts is not iterable then raise error.

    Returns
    -------
    H : Float
        The Hurst-index of this series.
    '''
    if not isinstance(ts, Iterable):
        raise ValueError("This sequence is not iterable !")
    ts = np.array(ts)
    # N is use for storge the length sequence
    N, RS, n = [], [], len(ts)
    while (True):
        N.append(n)
        # Calculate the average value of the series
        m = np.mean(ts)
        # Construct mean adjustment sequence
        mean_adj = ts - m
        # Construct cumulative deviation sequence
        cumulative_dvi = np.cumsum(mean_adj)
        # Calculate sequence range
        srange = max(cumulative_dvi) - min(cumulative_dvi)
        # Calculate the unbiased standard deviation of this sequence
        unbiased_std_dvi = np.std(ts)
        # Calculate the rescaled range of this sequence under n length
        RS.append(srange / unbiased_std_dvi)
        # While n < 4 then break
        if n < 4:
            break
        # Rebuild this sequence by half length
        ts, n = HalfSeries(ts, n)
    # Get Hurst-index by fit log(RS)~log(n)
    H = np.polyfit(np.log10(N), np.log10(RS), 1)[0]
    return H


def HalfSeries(s, n):
    '''
    if length(X) is odd:
        X <- {(X1 + X2) / 2, ..., (Xn-2 + Xn-1) / 2, Xn}
        n <- (n - 1) / 2
    else:
        X <- {(X1 + X2) / 2, ..., (Xn-1 + Xn) / 2}
        n <- n / 2
    return X, n
    '''
    X = []
    for i in range(0, len(s) - 1, 2):
        X.append((s[i] + s[i + 1]) / 2)
    # if length(s) is odd
    if len(s) % 2 != 0:
        X.append(s[-1])
        n = (n - 1) // 2
    else:
        n = n // 2
    return [np.array(X), n]


#########################################################

def dividing_data_set(date_set, node_feature, node_feature_value):
    """划分数据集"""
    # 先获取对应特征 node_feature 在数据集中所有条数据的有序取值数组
    feature_in_sets = date_set[node_feature]
    # 记录所有取值为 node_feature_value 数据编号
    reserved_group = [i for i in range(len(feature_in_sets)) if feature_in_sets[i] == node_feature_value]

    # 接着依据 reserved_group 中的组号保留属于当前分支的数据
    sub_date_set = {}
    for the_key in date_set:
        sub_date_set[the_key] = np.array([date_set[the_key][i] for i in reserved_group])

    # 最后，删除用过的特征列
    del (sub_date_set[node_feature])
    return sub_date_set


def gain(impurity_t, impurity_before_divide, data_set, probable_feature):
    """
    计算信息增益
    需要传入数据集划分前的不纯度、划分数据集所依赖特征对应的取值数组。考虑到在同一个节点测试不同子特征增益时都有用
    到划分前的不纯度，为了提升运行效率故在gain()外计算好该节点分裂前的不纯度后再传入gain()函数。其中数据集划分前的
    熵就是划分前的标签集labels的熵。其中按某特征划分后的不确定度，为依该特征各个取值划分的子数据集的中的标签集（即
    该特征划分完后所有的子标签集）的不确定度总和。

    Parameters
    ----------
    impurity_t:              str,不纯度的度量方式，只能是{"entropy","gini"}中的一个。
    impurity_before_divide:  float，表示数据集划分前的不纯度。
    data_set：               dict，划分前的数据集。
    probable_feature:        str，用于划分数据集的特征。

    Return
    ------
    result:      float，表征信息增益值。

    """
    impurity_after_divide = 0  # 初始化数据集划分后的不存度为0
    for value in set(data_set[probable_feature]):  # 获取该特征所有的取值并使用集合去重，遍历之
        one_sublabel_array = dividing_data_set(  # 获取该子数据集中的标签集数组
            date_set=data_set,
            node_feature=probable_feature,
            node_feature_value=value
        )['labels']
        impurity_after_divide = impurity(one_sublabel_array, impurity_t)  # 累加每个子数据标签集的不存度
    return impurity_before_divide - impurity_after_divide  # 做差得到这个特征的增益并返回


def gain_rate(impurity_t, impurity_before_divide, data_set, probable_feature):
    """
    计算信息增益率
    相对于信息增益的计算，信息增益率还要求解出由于该特征的不同取值带来的不确度。
     - 若由于特征取值带来的不确定度为0，说明无特征取值连续化影响，直接返回信息增益；
     - 若特征取值带来的不确定度不是0，则使用信息增益除以特征取证带来的不确定度。

    Parameters
    ----------
    impurity_t:              str,不纯度的度量方式，只能是{"entropy","gini"}中的一个。
    impurity_before_divide:  float，表示数据集划分前的不纯度。
    data_set：               dict，划分前的数据集。
    probable_feature:        str，用于划分数据集的特征。

    Return
    ------
    result:      float，表征信息增益值。

    """
    impurity_after_divide = 0  # 初始化数据集划分后的不存度为0
    for value in set(data_set[probable_feature]):  # 获取该特征所有的取值并使用集合去重，遍历之
        one_sublabel_array = dividing_data_set(  # 获取该子数据集中的标签集数组
            date_set=data_set,
            node_feature=probable_feature,
            node_feature_value=value
        )['labels']
    impurity_after_divide = impurity(one_sublabel_array, impurity_t)  # 累加每个子数据标签集的不存度
    gain = impurity_before_divide - impurity_after_divide  # 做差得到这个特征的增益

    feature_impurity = impurity(data_set[probable_feature], impurity_t)
    gain_rate = gain / feature_impurity if feature_impurity > 0 else gain
    return gain_rate


def impurity(anArray, impurity_t="entropy"):
    """
    计算混杂度

    Parameters
    ----------
    anArray：     an Array like object，由某特征依次对应每条数据下的取值构成。
    impurity_t：  str，表示混杂度的度量方式，只能是{"entropy","gini"}中的一个。

    Return
    result: float
        为计算得到的impurity的数值。
    """

    cnt = collections.Counter(anArray)
    data_length = len(anArray)
    pis = [cnt[i] / data_length for i in cnt]

    if impurity_t == "entropy":  # 信息熵：Entropy = -∑(i=1,n)|(pi·logpi)
        return -np.sum([pi * np.log2(pi) for pi in pis if pi > 0])
    elif impurity_t == "gini":  # 基尼系数：Gini = 1-∑(k=1,n)|(Pk)^2
        return 1 - np.sum([Pi ** 2 for Pi in pis])
    else:
        raise ValueError("impurity_t can only be one of {'entropy','gini'}")


def best_feature(impurity_t, date_set):
    """
    求取节点处的最佳特征

    Parameters
    ----------
    date_set:    dict，与某个节点处的对应的数据集

    Return
    ------
    result：     str，数据集date_set所属节点处可用于分裂的最佳特征
    """
    features = [i for i in date_set if i != "labels"]  # 获取数据集中当前节点处所有特征
    impurity_before_divide = impurity(date_set["labels"], impurity_t)  # 数据集划分前labels的混杂度
    max_gain_rate = -1  # 不会小于0，因此随便给个负数初始值
    the_best_feature = ""
    for probable_feature in features:
        rate = gain_rate(impurity_t, impurity_before_divide, date_set, probable_feature)
        print("%s:%f" % (probable_feature, rate))
        if rate > max_gain_rate:
            max_gain_rate = rate
            the_best_feature = probable_feature
    return the_best_feature


#########################################################

def load_data(csvFile, fields, yField):
    print('导入数据：' + csvFile)
    df = pd.read_csv(csvFile, low_memory=False)

    ###################### 样本数据概况 ############################
    # print('前5行数据：\n', df.head(5))
    # # 数据描述统计
    # print('数据维度', '-' * 30, 'n', df.shape)
    # # print('数据类型', '-' * 30, '\n', df.dtypes)
    # print('数据描述', '-' * 30, '\n', df.describe())
    # print('缺失值个数', '-' * 30, '\n', df.isnull().sum())
    # print('查看字段信息：', )
    # df.info()

    # # 箱型图，查看特征分布情况
    # X_tmp = df.drop(columns=[yField])
    # X_tmp.plot.box()
    # plt.show()

    # # 柱状图，查看因变量样本分布是否均衡
    # sns.countplot(x=yField, data=df)
    # plt.xlabel(yField);
    # plt.ylabel('Number of occurrences');
    # plt.show()

    # 特征预处理
    # print("特征预处理之前 n", df)
    # from sklearn import preprocessing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # for field in fields:
    #     df[field] = min_max_scaler.fit_transform(df[field].values.reshape(-1, 1))
    #     # print("特征预处理之后 n", df)

    # #################### 样本分析 ########################
    # print('相关分析')
    # fields.append(yField)
    # model_correlations(df[fields])
    # print('主成分分析')
    # model_PCA(df[fields])

    ################# 标签与数据集 #######
    # 标签
    labels = np.array(df[yField])  # y
    # 特征标签
    dataset = df[fields].values  # X

    # ################ 数据标准化 ####################
    # 数据均值-方差标准化
    features = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
    labels = labels.reshape((-1, 1))
    labels = (labels - labels.mean(axis=0)) / labels.std(axis=0)

    # # 数据标准化到0~1
    # features = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))
    # labels = labels.reshape((-1, 1))
    # labels = (labels - labels.min(axis=0)) / (labels.max(axis=0) - labels.min(axis=0))

    # features = dataset
    # labels = labels.reshape((-1, 1))

    # # 初始化特征的标准化器
    # ss = StandardScaler()
    # # 分别对数据的特征进行标准化处理
    # features = ss.fit_transform(dataset)
    # 转换为 np.array形式
    # features = np.array(dataset)

    return features, labels, df


def Residual(y_train, y_predict_test):
    # fieldValue = y_train.reshape(-1) - y_predict_test

    y_train_1d = y_train.reshape(-1) if y_train.ndim != 1 else y_train
    y_predict_test_1d = y_predict_test.reshape(-1) if y_predict_test.ndim != 1 else y_predict_test

    fieldValue = y_train_1d - y_predict_test_1d

    x = df['X'].values - 12121089
    y = df['Y'].values - 2835963
    
    gdf = gpd.GeoDataFrame(
        {'fieldValue': fieldValue},
        crs='EPSG:3857', 
        geometry=gpd.points_from_xy(x, y)
    )
    
    # 创建GeoDataFrame
    # gdf = gpd.GeoDataFrame(df, crs='EPSG:3857', geometry=gpd.points_from_xy(x, y))  # lng,lat根据实际字段选择
    print("创建权重矩阵...")
    # 根据需要选择合适的权重矩阵
    w = libpysal.weights.DistanceBand.from_dataframe(gdf, threshold=10000, binary=False)
    w.transform = 'r'

    print("计算 Moran's I指数...")
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


df = []
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    ##设置控制台显示格式
    np.set_printoptions(threshold=8)  # 显示所有行数据np.inf
    pd.set_option('display.max_columns', None)  # 显示所有列

    data_dir = r'F:\Map\base'
    os.chdir(data_dir)

    yField = 'probability'  # area_Proba, Fire_Count, probability
    fields = ['N_V_tmp', 'N_V_pre', 'N_V_wind_',
              'N_V_SLOP', 'N_V_DEM', 'N_V_ASPE',
              'YU_BI_DU', 'PINGJUN_XJ', 'N_V_GDP', 'N_V_POP',
              'N_road_dist', 'N_water_dist'

              # 'N_water_dist','N_road_dist','N_railways_d',
              # 'N_V_wet','N_V_dtr','N_V_pet','N_V_frs', #'N_V_tmx','N_V_tmn',
              # 'N_V_vap','N_V_srad_','N_V_lrad_','N_V_shum_','N_V_prec_',#'N_V_temp_','N_V_pres_',
              # 'N_V_residence_density', 'N_V_NDVI'
              ]

    # 近地面气温       temp    K         瞬时近地面（2m）气温
    # 地表气压         pres    Pa        瞬时地表气压
    # 近地面空气比湿    shum    kg/kg     瞬时近地面空气比湿
    # 近地面全风速      wind    m/s       瞬时近地面（风速仪高度）全风速
    # 向下短波辐射      srad    W/平方米   3小时平均（-1.5 hr～+1.5hr）向下短波辐射
    # 向下长波辐射      lrad    W/平方米   3小时平均（-1.5hr～+1.5hr）向下长波辐射。
    # 降水率           prec    mm/hr      3小时平均（-3.0hr～0.0hr）降水率。
    # # X, Y, k = [], [], 14
    # # for i in range(14, 501):
    # # X.append(i)
    # # Y.append(Hurst([random() for i in range(k)]))

    features, labels, df = load_data("modis_hn_Standardization.csv", fields,
                                     yField)  # modis_hn_70_Standardization

    # 共线性检验
    # fields = inflation_factor("modis_hn_70_Standardization.csv", fields)#modis_hn_70_spatial 'cof_Intercept',
    # inflation_factor(df, fields)  # modis_hn_70_spatial 'cof_Intercept',
    # # # draw_Data_histogram(df, 'Fire_Count')
    # exit(0)

    # # # 转换为数据集字典
    # # date_set = dict(zip(fields, features.T))
    # # date_set.update({"labels": labels.reshape(1,-1)[0]})  # 将标签集（labels，也就是输出y们）也加入数据集
    # # impurity_t = "entropy"  # 使用信息熵度量混杂度
    # # the_best_feature = best_feature(impurity_t, date_set)
    # # print("The best Feature:" + the_best_feature)
    # variation_index(np.transpose(features))

    #####################切分数据集， 训练集与测试集 #######
    # 将数据分为训练集和测试集 // 传入参数：特征、标签、比例、随机种子
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1000)

    # model_Ridge(X_train, y_train, X_test, y_test, fields)
    # model_Lasso(X_train, y_train, X_test, y_test, fields)
    # model_ElasticNet(X_train, y_train, X_test, y_test, fields)
    # model_BayesianRidge(X_train, y_train, X_test, y_test, fields)
    # model_Lars(X_train, y_train, X_test, y_test, fields)
    # model_LassoLars(X_train, y_train, X_test, y_test, fields)
    # model_PLSRegression(X_train, y_train, X_test, y_test, fields)

    # model_RandomForestRegressor(features, labels, X_test, y_test, fields)
    # model_AdaBoostRegressor(features, labels, X_test, y_test, fields)
    # model_SVR(features, labels, X_test, y_test, fields)
    # model_MLP(features, labels, X_test, y_test, fields)  
    model_KNeighborsRegressor(features, labels, X_test, y_test, fields)
    
    # model_LinearRegression(X_train, y_train, X_test, y_test, fields)
    # # model_polyfit(X_train, y_train, X_test, y_test, fields, 3)

    # model_KNeighborsRegressor(X_train, y_train, X_test, y_test, fields)
    # model_DecisionTreeRegressor(X_train, y_train, X_test, y_test, fields)
    # # model_MLP(X_train, y_train, X_test, y_test, fields)   

    # # model_ExtraTreesRegressor(X_train, y_train, X_test, y_test, fields)
    # # model_BaggingRegressor(X_train, y_train, X_test, y_test, fields)

    # model_RandomForestRegressor(X_train, y_train, X_test, y_test, fields)
    # model_AdaBoostRegressor(X_train, y_train, X_test, y_test, fields)
    # model_SVR(X_train, y_train, X_test, y_test, fields)  ###
    # exit(0)

    # model_LinearSVR(X_train, y_train, X_test, y_test, fields)  ###
    # model_GradientBoostingRegressor(X_train, y_train, X_test, y_test, fields) ####

    # y_train = [0 if lable < 0.5 else 1 for lable in y_train]
    # y_test = [0 if lable < 0.5 else 1 for lable in y_test]
    # model_LogisticRegression(X_train, y_train, X_test, y_test, fields)

    # model_DecisionTreeClassifier(X_train, y_train, X_test, y_test, fields)
    # model_RandomForestClassifier(X_train, y_train, X_test, y_test, fields)
