from typing import Union
import numpy as np
import pandas as pd
import multiprocessing as mp
from .kernel import GWRKernel, GTWRKernel
from .function import _compute_betas_gwr, surface_to_plane
from .obj import CalAicObj, CalMultiObj, BaseModel, GWRResults, GTWRResults, MGWRResults, MGTWRResults
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.neighbors import KDTree  # 确保导入KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
import gc


class GWR(BaseModel):
    """
    Geographically Weighted Regression
    """

    def __init__(
            self,
            coords: Union[np.ndarray, pd.DataFrame],
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame, pd.Series],
            bw: float,
            kernel: str = 'bisquare',
            fixed: bool = True,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False,
    ):
        """
        Parameters
        ----------
        coords        : array-like
                        n*2, spatial coordinates of the observations, if it's latitude and longitude,
                        the first column should be longitude

        X             : array-like
                        n*k, independent variable, excluding the constant

        y             : array-like
                        n*1, dependent variable

        bw            : scalar
                        bandwidth value consisting of either a distance or N
                        nearest neighbors; user specified or obtained using
                        sel

        kernel        : string
                        type of kernel function used to weight observations;
                        available options:
                        'gaussian'
                        'bisquare'
                        'exponential'

        fixed         : bool
                        True for distance based kernel function (default) and
                        False for adaptive (nearest neighbor) kernel function

        constant      : bool
                        True to include intercept (default) in model and False to exclude
                        intercept.

        thread        : int
                        The number of processes in parallel computation. If you have a large amount of data,
                        you can use it

        convert       : bool
                        Whether to convert latitude and longitude to plane coordinates.
        Examples
        --------
        import numpy as np
        from mgtwr.model import GWR
        np.random.seed(10)
        u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
        v = np.array([((i-1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
        t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
        x1 = np.random.uniform(0, 1, (1728, 1))
        x2 = np.random.uniform(0, 1, (1728, 1))
        epsilon = np.random.randn(1728, 1)
        beta0 = 5
        beta1 = 3 + (u + v + t)/6
        beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
        y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
        coords = np.hstack([u, v])
        X = np.hstack([x1, x2])
        gwr = GWR(coords, X, y, 0.8, kernel='gaussian', fixed=True).fit()
        print(gwr.R2)
        0.7128737240047688
        """
        super(GWR, self).__init__(X, y, kernel, fixed, constant)
        if thread < 1 or not isinstance(thread, int):
            raise ValueError('thread should be an integer greater than or equal to 1')
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.bw = bw
        self.thread = thread

    def _build_wi(self, i, bw):
        """
        calculate Weight matrix
        """
        try:
            gwr_kernel = GWRKernel(self.coords, bw, fixed=self.fixed, function=self.kernel)
            distance = gwr_kernel.cal_distance(i)
            wi = gwr_kernel.cal_kernel(distance)
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def cal_aic(self):
        """
        use for calculating AICc, BIC, CV and so on.
        """
        if self.thread > 1:
            result = list(zip(*Parallel(n_jobs=self.thread)(delayed(self._search_local_fit)(i) for i in range(self.n))))
        else:
            result = list(zip(*map(self._search_local_fit, range(self.n))))
        err2 = np.array(result[0]).reshape(-1, 1)
        hat = np.array(result[1]).reshape(-1, 1)
        aa = np.sum(err2 / ((1 - hat) ** 2))
        RSS = np.sum(err2)
        tr_S = np.sum(hat)
        llf = -np.log(RSS) * self.n / 2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2

        return CalAicObj(tr_S, float(llf), float(aa), self.n)

    def _search_local_fit(self, i):
        wi = self._build_wi(i, self.bw).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influx = np.dot(self.X[i], inv_xtx_xt[:, i])
        return reside * reside, influx

    def _local_fit(self, i):
        wi = self._build_wi(i, self.bw).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influx = np.dot(self.X[i], inv_xtx_xt[:, i])
        Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1)
        CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
        Si2 = np.sum(Si ** 2)
        return influx, reside, predict, betas.reshape(-1), CCT, Si2

    def _multi_fit(self, i):
        wi = self._build_wi(i, self.bw).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        pre = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - pre
        return betas.reshape(-1), pre, reside

    def cal_multi(self):
        """
        calculate betas, predict value and reside, use for searching best bandwidth in MGWR model by backfitting.
        """
        if self.thread > 1:
            result = list(zip(*Parallel(n_jobs=self.thread)(delayed(self._multi_fit)(i) for i in range(self.n))))
        else:
            result = list(zip(*map(self._multi_fit, range(self.n))))
        betas = np.array(result[0])
        pre = np.array(result[1]).reshape(-1, 1)
        reside = np.array(result[2]).reshape(-1, 1)
        return CalMultiObj(betas, pre, reside)

    def fit(self):
        """
        To fit GWR model
        """
        if self.thread > 1:
            result = list(zip(*Parallel(n_jobs=self.thread)(delayed(self._local_fit)(i) for i in range(self.n))))
        else:
            result = list(zip(*map(self._local_fit, range(self.n))))
        influ = np.array(result[0]).reshape(-1, 1)
        reside = np.array(result[1]).reshape(-1, 1)
        predict_value = np.array(result[2]).reshape(-1, 1)
        betas = np.array(result[3])
        CCT = np.array(result[4])
        tr_STS = np.array(result[5])
        return GWRResults(self, self.coords, self.X, self.y, self.bw, self.kernel, self.fixed,
                          influ, reside, predict_value, betas, CCT, tr_STS)

    # untest
    def predict(self, points, P, exog_scale=None, exog_resid=None, fit_params={}):
        """
        Method that predicts values of the dependent variable at un-sampled
        locations

        Parameters
        ----------
        points        : array-like
                        n*2, collection of n sets of (x,y) coordinates used for
                        calibration prediction locations
        P             : array
                        n*k, independent variables used to make prediction;
                        exlcuding the constant
        exog_scale    : scalar
                        estimated scale using sampled locations; defualt is None
                        which estimates a model using points from "coords"
        exog_resid    : array-like
                        estimated residuals using sampled locations; defualt is None
                        which estimates a model using points from "coords"; if
                        given it must be n*1 where n is the length of coords
        fit_params    : dict
                        key-value pairs of parameters that will be passed into fit
                        method to define estimation routine; see fit method for more details

        """
        if (exog_scale is None) & (exog_resid is None):
            train_gwr = self.fit(**fit_params)
            self.exog_scale = train_gwr.scale
            self.exog_resid = train_gwr.resid_response
        elif (exog_scale is not None) & (exog_resid is not None):
            self.exog_scale = exog_scale
            self.exog_resid = exog_resid
        else:
            raise InputError('exog_scale and exog_resid must both either be'
                             'None or specified')
        self.points = points
        if self.constant:
            P = np.hstack([np.ones((len(P), 1)), P])
            self.P = P
        else:
            self.P = P
        gwr = self.fit(**fit_params)

        return gwr


class MGWR(GWR):
    """
    Multiscale Geographically Weighted Regression
    """

    def __init__(
            self,
            coords: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            selector,
            kernel: str = 'bisquare',
            fixed: bool = False,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        """
        Parameters
        ----------
        coords        : array-like
                        n*2, spatial coordinates of the observations, if it's latitude and longitude,
                        the first column should be longitude

        X             : array-like
                        n*k, independent variable, excluding the constant

        y             : array-like
                        n*1, dependent variable

        selector      :SearchMGWRParameter object
                       valid SearchMGWRParameter that has successfully called
                       the "search" method. This parameter passes on
                       information from GAM model estimation including optimal
                       bandwidths.

        kernel        : string
                        type of kernel function used to weight observations;
                        available options:
                        'gaussian'
                        'bisquare'
                        'exponential'

        fixed         : bool
                        True for distance based kernel function (default) and  False for
                        adaptive (nearest neighbor) kernel function

        constant      : bool
                        True to include intercept (default) in model and False to exclude
                        intercept.

        thread        : int
                        The number of processes in parallel computation. If you have a large amount of data,
                        you can use it

        convert       : bool
                        Whether to convert latitude and longitude to plane coordinates.
        Examples
        --------
        import numpy as np
        from mgtwr.sel import SearchMGWRParameter
        from mgtwr.model import MGWR
        np.random.seed(10)
        u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
        v = np.array([((i-1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
        t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
        x1 = np.random.uniform(0, 1, (1728, 1))
        x2 = np.random.uniform(0, 1, (1728, 1))
        epsilon = np.random.randn(1728, 1)
        beta0 = 5
        beta1 = 3 + (u + v + t)/6
        beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
        y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
        coords = np.hstack([u, v])
        X = np.hstack([x1, x2])
        sel_multi = SearchMGWRParameter(coords, X, y, kernel='gaussian', fixed=True)
        bws = sel_multi.search(multi_bw_max=[40], verbose=True)
        mgwr = MGWR(coords, X, y, sel_multi, kernel='gaussian', fixed=True).fit()
        print(mgwr.R2)
        0.7045642214972343
        """
        self.selector = selector
        self.bws = self.selector.bws[0]  # final set of bandwidth
        self.bws_history = selector.bws[1]  # bws history in back_fitting
        self.betas = selector.bws[3]
        bw_init = self.selector.bws[5]  # initialization bandwidth
        super().__init__(
            coords, X, y, bw_init, kernel=kernel, fixed=fixed, constant=constant, thread=thread, convert=convert)
        self.n_chunks = None
        self.ENP_j = None

    def _chunk_compute(self, chunk_id=0):
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),
                       k), dtype='float32')  # partial R: n by chunk_size by k

        for i in range(n):
            wi = self._build_wi(i, self.bw).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  # n by chunk_size

        for iter_i in range(self.bws_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(
                                                                                   chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self._build_wi(index, self.bws_history[iter_i, j])
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1)) ** 2).sum(
                axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        return ENP_j, CCT,

    def fit(self, n_chunks: int = 1, skip_calculate: bool = False):
        """
        Compute MGWR inference by chunk to reduce memory footprint.
        Parameters
        ----------
        n_chunks       : int
                         divided into n_chunks steps to reduce memory consumption
        skip_calculate : bool
                         if True, skip calculate CCT, ENP and other variables derived from it
        """
        pre = np.sum(self.X * self.betas, axis=1).reshape(-1, 1)
        ENP_j = None
        CCT = None
        if not skip_calculate:
            self.n_chunks = n_chunks
            result = map(self._chunk_compute, (range(n_chunks)))
            result_list = list(zip(*result))
            ENP_j = np.sum(np.array(result_list[0]), axis=0)
            CCT = np.sum(np.array(result_list[1]), axis=0)
        return MGWRResults(self,
                           self.coords, self.X, self.y, self.bws, self.kernel, self.fixed,
                           self.bws_history, self.betas, pre, ENP_j, CCT)


class GTWR(BaseModel):
    """
    Geographically and Temporally Weighted Regression

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observations

    t             : array-like
                    n*1, time location

    X             : array-like
                        n*k, independent variable, excluding the constant

    y             : array-like
                    n*1, dependent variable

    bw            : scalar
                    bandwidth value consisting of either a distance or N
                    nearest neighbors; user specified or obtained using
                    sel

    tau           : scalar
                    spatio-temporal scale

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : bool
                    True for distance based kernel function (default) and
                    False for adaptive (nearest neighbor) kernel function

    constant      : bool
                    True to include intercept (default) in model and False to exclude
                    intercept.

    Examples
    --------
    import numpy as np
    from mgtwr.model import GTWR
    np.random.seed(10)
    u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
    v = np.array([((i-1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
    t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
    x1 = np.random.uniform(0, 1, (1728, 1))
    x2 = np.random.uniform(0, 1, (1728, 1))
    epsilon = np.random.randn(1728, 1)
    beta0 = 5
    beta1 = 3 + (u + v + t)/6
    beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
    coords = np.hstack([u, v])
    X = np.hstack([x1, x2])
    gtwr = GTWR(coords, t, X, y, 0.8, 1.9, kernel='gaussian', fixed=True).fit()
    print(gtwr.R2)
    0.9899869616636376
    """

    def __init__(
            self,
            coords: Union[np.ndarray, pd.DataFrame],
            t: Union[np.ndarray, pd.DataFrame],
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame],
            bw: float,
            tau: float,
            kernel: str = 'gaussian',
            fixed: bool = False,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        super(GTWR, self).__init__(X, y, kernel, fixed, constant)
        if thread < 1 or not isinstance(thread, int):
            raise ValueError('thread should be an integer greater than or equal to 1')
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.t = t
        self.bw = bw
        self.tau = tau
        self.bw_s = self.bw
        self.bw_t = np.sqrt(self.bw ** 2 / self.tau)
        self.thread = thread

    def _build_wi(self, i, bw, tau):
        """
        calculate Weight matrix
        """
        try:
            gtwr_kernel = GTWRKernel(self.coords, self.t, bw, tau, fixed=self.fixed, function=self.kernel)
            distance = gtwr_kernel.cal_distance(i)
            wi = gtwr_kernel.cal_kernel(distance)
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def cal_aic(self):

        """
        use for calculating AICc, BIC, CV and so on.
        """
        if self.thread > 1:
            result = list(zip(*Parallel(n_jobs=self.thread)(delayed(self._search_local_fit)(i) for i in range(self.n))))
        else:
            result = list(zip(*map(self._search_local_fit, range(self.n))))
        err2 = np.array(result[0]).reshape(-1, 1)
        hat = np.array(result[1]).reshape(-1, 1)
        aa = np.sum(err2 / ((1 - hat) ** 2))
        RSS = np.sum(err2)
        tr_S = np.sum(hat)
        llf = -np.log(RSS) * self.n / 2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2

        return CalAicObj(tr_S, float(llf), float(aa), self.n)

    def _search_local_fit(self, i):
        wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
        betas, xtx_inv_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influ = np.dot(self.X[i], xtx_inv_xt[:, i])
        return reside * reside, influ

    def _local_fit(self, i):
        wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
        betas, xtx_inv_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influ = np.dot(self.X[i], xtx_inv_xt[:, i])
        Si = np.dot(self.X[i], xtx_inv_xt).reshape(-1)
        CCT = np.diag(np.dot(xtx_inv_xt, xtx_inv_xt.T)).reshape(-1)
        Si2 = np.sum(Si ** 2)
        return influ, reside, predict, betas.reshape(-1), CCT, Si2

    def _multi_fit(self, i):
        wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        pre = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - pre
        return betas.reshape(-1), pre, reside

    def cal_multi(self):
        """
        calculate betas, predict value and reside, use for searching best bandwidth in MGWR model by backfitting.
        """
        if self.thread > 1:
            result = list(zip(*Parallel(n_jobs=self.thread)(delayed(self._multi_fit)(i) for i in range(self.n))))
        else:
            result = list(zip(*map(self._multi_fit, range(self.n))))
        betas = np.array(result[0])
        pre = np.array(result[1]).reshape(-1, 1)
        reside = np.array(result[2]).reshape(-1, 1)
        return CalMultiObj(betas, pre, reside)

    def fit(self):
        """
        fit GTWR models

        """
        if self.thread > 1:
            result = list(zip(*Parallel(n_jobs=self.thread)(delayed(self._local_fit)(i) for i in range(self.n))))
        else:
            result = list(zip(*map(self._local_fit, range(self.n))))
        influ = np.array(result[0]).reshape(-1, 1)
        reside = np.array(result[1]).reshape(-1, 1)
        predict_value = np.array(result[2]).reshape(-1, 1)
        betas = np.array(result[3])
        CCT = np.array(result[4])
        tr_STS = np.array(result[5])
        model = self
        return GTWRResults(model,
                           self.coords, self.t, self.X, self.y, self.bw, self.tau, self.kernel, self.fixed,
                           influ, reside, predict_value, betas, CCT, tr_STS
                           )

    def cal_weights(self):
        """
        计算GTWR模型的时空权重矩阵

        Returns
        -------
        weights_matrix : array-like
                         n*n 权重矩阵，其中weights_matrix[i,j]表示第j个观测点对第i个观测点的时空权重
        """
        n = self.n  # 获取观测点数量
        weights_matrix = np.zeros((n, n))

        # 对每个观测点i，计算其对所有其他观测点的时空权重
        for i in range(n):
            # 调用已有的_build_wi函数计算时空权重
            wi = self._build_wi(i, self.bw, self.tau).reshape(-1)
            weights_matrix[i, :] = wi

        return weights_matrix

    # 测试
    def predict(self, X_new, coords_new, t_new=None):

        """
        对新数据点进行GTWR预测

        Parameters
        ----------
        X_new : array-like
                n_new*k, 新数据的自变量值，不包含常数项

        coords_new : array-like
                     n_new*2, 新数据点的空间坐标(x,y)

        t_new : array-like, 可选
                n_new*1, 新数据点的时间坐标。如果未提供，则使用训练数据的时间范围的中间值

        Returns
        -------
        predictions : array-like
                      n_new*1, 预测值

        confidence_intervals : array-like, 可选
                               n_new*2, 95%置信区间的上下界（如果available）
        """
        # 确保输入数据是numpy数组
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values
        if isinstance(coords_new, pd.DataFrame):
            coords_new = coords_new.values
        if t_new is not None:
            if isinstance(t_new, pd.DataFrame):
                t_new = t_new.values
        else:
            # 如果未提供时间，则使用训练数据时间的中间值
            train_time_mid = np.median(self.t)
            t_new = np.full((X_new.shape[0], 1), train_time_mid)

        n_new = X_new.shape[0]
        predictions = np.zeros(n_new)

        # 如果模型已拟合，直接使用现有参数
        if hasattr(self, 'results') and self.results is not None:
            # 使用已拟合模型的参数进行预测
            betas = self.results.betas  # 获取已估计的系数

            # 为每个新数据点找到最近的训练点，并使用其系数进行预测
            for i in range(n_new):
                # 计算与所有训练点的时空距离
                dists_s = np.sum((self.coords - coords_new[i]) ** 2, axis=1) ** 0.5
                dists_t = np.abs(self.t.flatten() - t_new[i, 0])
                dists_st = np.sqrt(dists_s ** 2 + (dists_t / self.tau) ** 2)

                # 找到最近的训练点
                nearest_idx = np.argmin(dists_st)

                # 使用最近训练点的系数进行预测
                if self.constant:
                    # 如果模型包含常数项，添加一个1到X的前面
                    pred_X = np.hstack([np.ones((1, 1)), X_new[i:i + 1, :]])
                else:
                    pred_X = X_new[i:i + 1, :]

                predictions[i] = np.sum(pred_X * betas[nearest_idx])

        else:
            # 如果模型尚未拟合，则对每个新点单独计算局部回归
            for i in range(n_new):
                # 构建该点的时空权重
                wi = self._build_wi_new(coords_new[i], t_new[i, 0]).reshape(-1, 1)

                # 计算局部系数
                betas, _ = _compute_betas_gwr(self.y, self.X, wi)

                # 进行预测
                if self.constant:
                    # 如果模型包含常数项，添加一个1到X的前面
                    pred_X = np.hstack([np.ones((1, 1)), X_new[i:i + 1, :]])
                else:
                    pred_X = X_new[i:i + 1, :]

                predictions[i] = np.sum(pred_X * betas)

        return predictions.reshape(-1, 1)

    # 测试
    def _build_wi_new(self, coord_new, t_new):
        """为新数据点构建时空权重"""
        # 计算新点到所有训练点的时空距离
        dists_s = np.sum((self.coords - coord_new) ** 2, axis=1) ** 0.5
        dists_t = np.abs(self.t.flatten() - t_new)

        # 组合时空距离
        dists_st = np.sqrt(dists_s ** 2 + (dists_t / self.tau) ** 2)

        # 使用核函数计算权重
        gtwr_kernel = GTWRKernel(self.coords, self.t, self.bw, self.tau,
                                 fixed=self.fixed, function=self.kernel)
        wi = gtwr_kernel.cal_kernel(dists_st)

        return wi


class GTWRF(GTWR):
    def __init__(
            self, coords, t, X, y, bw=None, tau=None, rf_params=None,
            kernel='gaussian', fixed=False, gtwr_model=None, weight_threshold=1e-2, thread: int = 1,
            constant: bool = True, min_neighbors: int = 1
    ):
        super(GTWRF, self).__init__(coords, t, X, y, bw, tau, kernel, fixed, constant, thread)
        if thread < 1 or not isinstance(thread, int):
            raise ValueError('thread should be an integer greater than or equal to 1')

        # 验证min_neighbors参数
        if min_neighbors < 1:
            raise ValueError('min_neighbors should be at least 1')
        self.min_neighbors = min_neighbors  # 最少邻居点数

        self.X = self._convert_to_numpy(X).astype(np.float32)
        self.y = self._convert_to_numpy(y).flatten().astype(np.float32)
        self.coords = self._convert_to_numpy(coords).astype(np.float32)
        self.t = self._convert_to_numpy(t).astype(np.float32) if t is not None else np.zeros((len(self.coords), 1),
                                                                                             dtype=np.float32)
        self.weight_threshold=weight_threshold
        
        if gtwr_model:
            self.bw = gtwr_model.bw
            self.tau = gtwr_model.tau
            self.kernel = gtwr_model.kernel
            self.fixed = gtwr_model.fixed
        else:
            self.bw = bw
            self.tau = tau
            self.kernel = kernel
            self.fixed = fixed
            
        self.rf_params = rf_params or {}
        self.predict_values = None
        self.feature_importances = None
        self.thread = thread
        self.tree = None

    def _init_kdtree(self):
        """初始化KDTree用于邻居搜索"""
        self.tree = KDTree(self.coords)

    def _build_wi(self, i, bw, tau):
        """计算权重矩阵时保持float32类型"""
        try:
            gtwr_kernel = GTWRKernel(self.coords, self.t, bw, tau, fixed=self.fixed, function=self.kernel)
            distance = gtwr_kernel.cal_distance(i).astype(np.float32)
            wi = gtwr_kernel.cal_kernel(distance).astype(np.float32)
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def fit(self):
        print(f"有效训练点数量: {len(self.coords)}")
        # results = Parallel(n_jobs=self.thread)(
        #     delayed(self._fit_single_default)(idx)
        #     for idx in tqdm(range(len(self.coords)), desc="Training Local Models")
        # )

        results = []
        for idx in tqdm(range(len(self.coords)), desc="Training Local Models"):
            results.append(self._fit_single_default(idx))

        # 提取预测值和特征重要性
        self.predict_values = np.zeros(len(self.coords), dtype=np.float32).reshape(-1, 1)
        self.feature_importances = np.zeros((len(self.coords), self.X.shape[1]), dtype=np.float32)

        for idx, pred_val, feat_imp in results:
            self.predict_values[idx] = pred_val
            self.feature_importances[idx] = feat_imp

        # 计算结果时保持float32类型
        print(f"计算结果生成")
        influ, reside, CCT, tr_STS = self._calculate_gtwp_results()

        model = self
        
        # return GTWRResults(model,
                           # self.coords, self.t, self.X, self.y, self.bw, self.tau, self.kernel, self.fixed,
                           # influ, reside, self.predict_values.reshape(-1, 1), self.feature_importances, CCT, tr_STS
                           # )
        return GTWRResults(model,
                           self.coords, self.t, self.X, self.y, self.bw, self.tau, self.kernel, self.fixed,
                           influ, reside, self.predict_values, self.feature_importances, CCT, tr_STS
                           )

    def _fit_single_default(self, idx):
        """修改：只返回预测值和特征重要性，不存储完整模型"""
        weights = self._build_wi(idx, self.bw, self.tau).astype(np.float32)
        valid_mask = weights > self.weight_threshold
        valid_X = self.X[valid_mask]
        valid_y = self.y[valid_mask]
        valid_weights = weights[valid_mask].flatten()

        valid_neighbors = np.sum(valid_mask)

        if valid_neighbors < self.min_neighbors:
            print(f"点{idx}邻居数不足({valid_neighbors}), 使用全局模型")
            model = RandomForestRegressor(**self.rf_params)
            model.fit(self.X, self.y)
        else:
            model = RandomForestRegressor(**self.rf_params)
            model.fit(valid_X, valid_y, sample_weight=valid_weights)

        # 计算当前点的预测值
        X_i = self.X[idx:idx + 1].reshape(1, -1).astype(np.float32)
        predict_value = model.predict(X_i)[0].astype(np.float32)
        
        # 获取特征重要性
        feature_importance = model.feature_importances_.astype(np.float32)
        
        # 释放模型内存
        del model, valid_mask, valid_weights, valid_X, valid_y, weights
        gc.collect()

        return idx, predict_value, feature_importance

    def _calculate_gtwp_results(self):
        """修改：移除对local_models的依赖，使用已存储的predict_values"""
        n = len(self.coords)
        n_features = self.X.shape[1]

        influ = np.zeros(n, dtype=np.float32)
        reside = np.zeros(n, dtype=np.float32)
        CCT = np.zeros((n, n_features), dtype=np.float32)
        tr_STS = np.zeros(n, dtype=np.float32)

        for idx in range(n):
            y_i = self.y[idx].astype(np.float32)
            
            # 使用已存储的预测值
            predict_value = self.predict_values[idx]
            reside[idx] = (y_i - predict_value).astype(np.float32)

            weights = self._build_wi(idx, self.bw, self.tau).astype(np.float32)
            W_i = weights.reshape(-1, 1).astype(np.float32)
            
            try:
                xt = (self.X * W_i).T.astype(np.float32)
                xtx = np.dot(xt, self.X).astype(np.float32)
                xtx_inv_xt = np.linalg.solve(xtx, xt).astype(np.float32)
                influ[idx] = np.dot(self.X[idx], xtx_inv_xt[:, idx]).astype(np.float32)

                CCT[idx] = np.diag(np.dot(xtx_inv_xt, xtx_inv_xt.T)).reshape(-1).astype(np.float32)
                si = np.dot(self.X[idx], xtx_inv_xt).reshape(-1).astype(np.float32)
                tr_STS[idx] = np.sum(si ** 2).astype(np.float32)
            except:
                influ[idx] = 0.0
                CCT[idx] = 0.0
                tr_STS[idx] = 0.0

        return influ.reshape(-1, 1), reside.reshape(-1, 1), CCT, tr_STS.reshape(-1, 1)

    def predict(self, X_new, coords_new, t_new):
        """修改：预测逻辑需要调整，因为不再存储完整模型"""
        X_new_np = self._convert_to_numpy(X_new).astype(np.float32)
        coords_new_np = self._convert_to_numpy(coords_new).astype(np.float32)
        t_new_np = self._convert_to_numpy(t_new).astype(np.float32)

        n_new = X_new_np.shape[0]
        predictions = np.zeros(n_new, dtype=np.float32)

        if self.tree is None:
            self._init_kdtree()

        for i in range(n_new):
            dists, indices = self.tree.query(coords_new_np[i:i + 1], k=self.min_neighbors + 1)
            valid_neighbors = np.sum(dists[0, 1:] > 0)

            if valid_neighbors >= self.min_neighbors:
                nearest_idx = indices[0, np.argmin(dists[0, 1:]) + 1]
                
                # 使用最近点的特征重要性重新训练模型
                weights = self._build_wi(nearest_idx, self.bw, self.tau).astype(np.float32)
                valid_mask = weights > self.weight_threshold
                valid_X = self.X[valid_mask]
                valid_y = self.y[valid_mask]
                valid_weights = weights[valid_mask].flatten()

                model = RandomForestRegressor(**self.rf_params)
                model.fit(valid_X, valid_y, sample_weight=valid_weights)
                
                if isinstance(X_new, pd.DataFrame):
                    pred = model.predict(X_new.iloc[[i]])[0].astype(np.float32)
                else:
                    pred = model.predict(X_new_np[i:i + 1])[0].astype(np.float32)
                
                predictions[i] = pred
                
                # 释放临时模型
                del model, valid_mask, valid_weights, valid_X, valid_y, weights
                gc.collect()
            else:
                predictions[i] = np.mean(self.y)
                print(f"预测点{i}邻居数不足({valid_neighbors}), 使用全局平均值")

        return predictions.reshape(-1, 1)

    def _convert_to_numpy(self, data):
        # 核心优化点2：确保转换后的数据类型统一
        arr = data.values if hasattr(data, 'values') else np.array(data)
        return arr.astype(np.float32)  # 新增类型转换

    def _calculate_model_metrics(self):
        y_pred = np.zeros_like(self.y, dtype=np.float32)
        for idx, model in self.local_models.items():
            y_pred[idx] = model.predict([self.X[idx]])[0].astype(np.float32)

        residuals = (self.y - y_pred).astype(np.float32)
        self.r2 = 1 - (np.sum(residuals ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)).astype(np.float32)
        self.mae = np.mean(np.abs(residuals)).astype(np.float32)
        self.mse = np.mean(residuals ** 2).astype(np.float32)
        self.rmse = np.sqrt(self.mse).astype(np.float32)

        hat_matrix_diag = np.zeros(len(self.coords), dtype=np.float32)
        for i in range(len(self.coords)):
            X_i = self.X[i:i + 1].reshape(1, -1).astype(np.float32)
                        
            weights = self._build_wi(idx, self.bw, self.tau).astype(np.float32)  # 权重矩阵保持float32
            # 杠杆值计算
            W_i = np.diag(weights.reshape(-1, 1).astype(np.float32))  # 权重矩阵转换为对角矩阵时保持类型

            XWX = X_i.T @ W_i @ X_i
            XWX_inv = np.linalg.pinv(XWX).astype(np.float32)
            hat_matrix_diag[i] = (X_i @ XWX_inv @ X_i.T).item().astype(np.float32)

        self.edf = np.sum(hat_matrix_diag).astype(np.float32)
        self.aic = 2 * self.edf + len(self.coords) * np.log(np.sum(residuals ** 2) / len(self.coords)).astype(
            np.float32)


class GTWRF1(GTWR):
    def __init__(
            self, coords, t, X, y, bw=None, tau=None, rf_params=None,
            kernel='gaussian', fixed=False, gtwr_model=None, weight_threshold=1e-2, thread: int = 1,
            constant: bool = True, min_neighbors: int = 1  # 新增参数
    ):
        super(GTWRF, self).__init__(coords, t, X, y, bw, tau, kernel, fixed, constant, thread)
        if thread < 1 or not isinstance(thread, int):
            raise ValueError('thread should be an integer greater than or equal to 1')

        # 验证min_neighbors参数
        if min_neighbors < 1:
            raise ValueError('min_neighbors should be at least 1')
        self.min_neighbors = min_neighbors  # 最少邻居点数

        self.X = self._convert_to_numpy(X).astype(np.float32)
        self.y = self._convert_to_numpy(y).flatten().astype(np.float32)
        self.coords = self._convert_to_numpy(coords).astype(np.float32)
        self.t = self._convert_to_numpy(t).astype(np.float32) if t is not None else np.zeros((len(self.coords), 1),
                                                                                             dtype=np.float32)
        self.weight_threshold=weight_threshold
        
        if gtwr_model:
            self.bw = gtwr_model.bw
            self.tau = gtwr_model.tau
            self.kernel = gtwr_model.kernel
            self.fixed = gtwr_model.fixed
        else:
            self.bw = bw
            self.tau = tau
            self.kernel = kernel
            self.fixed = fixed
            
        self.rf_params = rf_params or {}
        self.local_models = {}
        self.thread = thread
        self.tree = None

    def _init_kdtree(self):
        """初始化KDTree用于邻居搜索"""
        self.tree = KDTree(self.coords)

    def _build_wi(self, i, bw, tau):
        """计算权重矩阵时保持float32类型"""
        try:
            gtwr_kernel = GTWRKernel(self.coords, self.t, bw, tau, fixed=self.fixed, function=self.kernel)
            distance = gtwr_kernel.cal_distance(i).astype(np.float32)
            wi = gtwr_kernel.cal_kernel(distance).astype(np.float32)
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def fit(self):
        print(f"有效训练点数量: {len(self.coords)}")
        # results = Parallel(n_jobs=self.thread)(
            # delayed(self._fit_single_default)(idx)
            # for idx in tqdm(range(len(self.coords)), desc="Training Local Models")
        # )

        results = []
        for idx in tqdm(range(len(self.coords)), desc="Training Local Models"):
            results.append(self._fit_single_default(idx))

        self.local_models = {idx: model for idx, model in results}

        # 计算结果时保持float32类型
        print(f"计算结果生成")
        influ, reside, predict_value, betas, CCT, tr_STS = self._calculate_gtwp_results()

        model = self
        
        return GTWRResults(model,
                           self.coords, self.t, self.X, self.y, self.bw, self.tau, self.kernel, self.fixed,
                           influ, reside, predict_value, betas, CCT, tr_STS
                           )

    def _fit_single_default(self, idx):
        """带邻居检查和动态权重筛选的局部模型训练"""
        # with np.errstate(over='ignore', invalid='ignore'):
        
        weights = self._build_wi(idx, self.bw, self.tau).astype(np.float32)  # 权重矩阵保持float32
        # 筛选有效权重及其对应的索引
        valid_mask = weights > self.weight_threshold
        valid_X = self.X[valid_mask]
        valid_y = self.y[valid_mask]
        valid_weights = weights[valid_mask].flatten()

        valid_neighbors = np.sum(valid_mask)

        if valid_neighbors < self.min_neighbors:
            # 邻居数不足时，尝试降低阈值或使用全局模型
            print(f"点{idx}邻居数不足({valid_neighbors}), 使用全局模型")
            model = RandomForestRegressor(**self.rf_params)
            model.fit(self.X, self.y)
        else:
            # print(f"点{idx}使用{valid_neighbors}个有效邻居训练局部模型(阈值={weight_threshold})")
            model = RandomForestRegressor(**self.rf_params)
            model.fit(valid_X, valid_y, sample_weight=valid_weights)

        # 任务结束前释放资源
        del valid_mask, valid_weights, valid_X, valid_y, weights
        gc.collect()

        return idx, model

    def _calculate_gtwp_results(self):
        n = len(self.coords)
        n_features = self.X.shape[1]

        # 核心优化点4：结果数组初始化时指定float32
        influ = np.zeros(n, dtype=np.float32)
        reside = np.zeros(n, dtype=np.float32)
        predict_value = np.zeros(n, dtype=np.float32)
        CCT = np.zeros((n, n_features), dtype=np.float32)
        tr_STS = np.zeros(n, dtype=np.float32)
        betas = np.zeros((n, n_features), dtype=np.float32)

        for idx in range(n):
            X_i = self.X[idx:idx + 1].reshape(1, -1).astype(np.float32)
            y_i = self.y[idx].astype(np.float32)
            model = self.local_models[idx]

            # 预测值与残差保持float32
            predict_value[idx] = model.predict(X_i)[0].astype(np.float32)
            reside[idx] = (y_i - predict_value[idx]).astype(np.float32)

            weights = self._build_wi(idx, self.bw, self.tau).astype(np.float32)  # 权重矩阵保持float32
            # 杠杆值计算
            W_i = weights.reshape(-1, 1).astype(np.float32)
            try:
                xt = (self.X * W_i).T.astype(np.float32)
                xtx = np.dot(xt, self.X).astype(np.float32)
                xtx_inv_xt = np.linalg.solve(xtx, xt).astype(np.float32)
                influ[idx] = np.dot(self.X[idx], xtx_inv_xt[:, idx]).astype(np.float32)

                # 条件数迹与迹统计量
                CCT[idx] = np.diag(np.dot(xtx_inv_xt, xtx_inv_xt.T)).reshape(-1).astype(np.float32)
                si = np.dot(self.X[idx], xtx_inv_xt).reshape(-1).astype(np.float32)
                tr_STS[idx] = np.sum(si ** 2).astype(np.float32)
            except:
                influ[idx] = 0.0
                CCT[idx] = 0.0
                tr_STS[idx] = 0.0

            # 特征重要性保持float32
            betas[idx] = model.feature_importances_.astype(np.float32)

        return influ.reshape(-1, 1), reside.reshape(-1, 1), predict_value.reshape(-1, 1), betas, CCT, tr_STS.reshape(-1,
                                                                                                                     1)

    def predict(self, X_new, coords_new, t_new):
        X_new_np = self._convert_to_numpy(X_new).astype(np.float32)
        coords_new_np = self._convert_to_numpy(coords_new).astype(np.float32)
        t_new_np = self._convert_to_numpy(t_new).astype(np.float32)

        n_new = X_new_np.shape[0]
        predictions = np.zeros(n_new, dtype=np.float32)

        # 确保KDTree已初始化
        if self.tree is None:
            self._init_kdtree()

        for i in range(n_new):
            # 搜索最近的min_neighbors+1个点
            dists, indices = self.tree.query(coords_new_np[i:i + 1], k=self.min_neighbors + 1)
            valid_neighbors = np.sum(dists[0, 1:] > 0)  # 排除自身

            if valid_neighbors >= self.min_neighbors:
                # 找到最近的有效邻居点
                nearest_idx = indices[0, np.argmin(dists[0, 1:]) + 1]  # 排除自身
                model = self.local_models.get(nearest_idx)

                if model:
                    if isinstance(X_new, pd.DataFrame):
                        pred = model.predict(X_new.iloc[[i]])[0].astype(np.float32)
                    else:
                        pred = model.predict(X_new_np[i:i + 1])[0].astype(np.float32)
                    predictions[i] = pred
                else:
                    # 模型不存在时使用默认预测
                    predictions[i] = np.mean(self.y)
            else:
                # 邻居数不足时使用全局平均
                predictions[i] = np.mean(self.y)
                print(f"预测点{i}邻居数不足({valid_neighbors}), 使用全局平均值")

        return predictions.reshape(-1, 1)

    def _convert_to_numpy(self, data):
        # 核心优化点2：确保转换后的数据类型统一
        arr = data.values if hasattr(data, 'values') else np.array(data)
        return arr.astype(np.float32)  # 新增类型转换

    def _calculate_model_metrics(self):
        y_pred = np.zeros_like(self.y, dtype=np.float32)
        for idx, model in self.local_models.items():
            y_pred[idx] = model.predict([self.X[idx]])[0].astype(np.float32)

        residuals = (self.y - y_pred).astype(np.float32)
        self.r2 = 1 - (np.sum(residuals ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)).astype(np.float32)
        self.mae = np.mean(np.abs(residuals)).astype(np.float32)
        self.mse = np.mean(residuals ** 2).astype(np.float32)
        self.rmse = np.sqrt(self.mse).astype(np.float32)

        hat_matrix_diag = np.zeros(len(self.coords), dtype=np.float32)
        for i in range(len(self.coords)):
            X_i = self.X[i:i + 1].reshape(1, -1).astype(np.float32)
                        
            weights = self._build_wi(idx, self.bw, self.tau).astype(np.float32)  # 权重矩阵保持float32
            # 杠杆值计算
            W_i = np.diag(weights.reshape(-1, 1).astype(np.float32))  # 权重矩阵转换为对角矩阵时保持类型

            XWX = X_i.T @ W_i @ X_i
            XWX_inv = np.linalg.pinv(XWX).astype(np.float32)
            hat_matrix_diag[i] = (X_i @ XWX_inv @ X_i.T).item().astype(np.float32)

        self.edf = np.sum(hat_matrix_diag).astype(np.float32)
        self.aic = 2 * self.edf + len(self.coords) * np.log(np.sum(residuals ** 2) / len(self.coords)).astype(
            np.float32)


class MGTWR(GTWR):
    """
    Multiscale GTWR estimation and inference.

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observatons

    t             : array
                    n*1, time location

    X             : array-like
                        n*k, independent variable, excluding the constant

    y             : array-like
                    n*1, dependent variable

    selector      : SearchMGTWRParameter object
                    valid SearchMGTWRParameter object that has successfully called
                    the "search" method. This parameter passes on
                    information from GAM model estimation including optimal
                    bandwidths.

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : bool
                    True for distance based kernel function (default) and  False for
                    adaptive (nearest neighbor) kernel function

    constant      : bool
                    True to include intercept (default) in model and False to exclude
                    intercept.
    Examples
    --------
    import numpy as np
    from mgtwr.sel import SearchMGTWRParameter
    from mgtwr.model import MGTWR
    np.random.seed(10)
    u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
    v = np.array([((i-1) % 144)//12 for i in range(1, 1729)]).reshape(-1, 1)
    t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
    x1 = np.random.uniform(0, 1, (1728, 1))
    x2 = np.random.uniform(0, 1, (1728, 1))
    epsilon = np.random.randn(1728, 1)
    beta0 = 5
    beta1 = 3 + (u + v + t)/6
    beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
    coords = np.hstack([u, v])
    X = np.hstack([x1, x2])
    sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
    bws = sel_multi.search(multi_bw_min=[0.1], verbose=True, tol_multi=1.0e-4)
    mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel='gaussian', fixed=True).fit()
    print(mgtwr.R2)
    0.9972924820674222
    """

    def __init__(
            self,
            coords: np.ndarray,
            t: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            selector,
            kernel: str = 'bisquare',
            fixed: bool = False,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        self.selector = selector
        self.bws = self.selector.bws[0]  # final set of bandwidth
        self.taus = self.selector.bws[1]
        self.bw_ts = np.sqrt(self.bws ** 2 / self.taus)
        self.bws_history = selector.bws[2]  # bws history in back_fitting
        self.taus_history = selector.bws[3]
        self.betas = selector.bws[5]
        bw_init = self.selector.bws[7]  # initialization bandwidth
        tau_init = self.selector.bws[8]
        super().__init__(coords, t, X, y, bw_init, tau_init,
                         kernel=kernel, fixed=fixed, constant=constant, thread=thread, convert=convert)
        self.n_chunks = None
        self.ENP_j = None

    def _chunk_compute(self, chunk_id=0):
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),
                       k))  # partial R: n by chunk_size by k

        for i in range(n):
            wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  # n by chunk_size

        for iter_i in range(self.bws_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(
                                                                                   chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self._build_wi(index, self.bws_history[iter_i, j],
                                            self.taus_history[iter_i, j])
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1)) ** 2).sum(
                axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        return ENP_j, CCT,

    def fit(self, n_chunks: int = 1, skip_calculate: bool = False):
        """
        Compute MGTWR inference by chunk to reduce memory footprint.
        Parameters
        ----------
        n_chunks       : int
                         divided into n_chunks steps to reduce memory consumption
        skip_calculate : bool
                         if True, skip calculate CCT, ENP and other variables derived from it
        """
        pre = np.sum(self.X * self.betas, axis=1).reshape(-1, 1)
        ENP_j = None
        CCT = None
        if not skip_calculate:
            self.n_chunks = n_chunks
            result = map(self._chunk_compute, (range(n_chunks)))
            result_list = list(zip(*result))
            ENP_j = np.sum(np.array(result_list[0]), axis=0)
            CCT = np.sum(np.array(result_list[1]), axis=0)
        return MGTWRResults(self,
                            self.coords, self.t, self.X, self.y, self.bws, self.taus, self.kernel, self.fixed,
                            self.bw_ts,
                            self.bws_history, self.taus_history, self.betas, pre, ENP_j, CCT)
