import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import t


class CalAicObj:

    def __init__(self, tr_S, llf, aa, n):
        self.tr_S = tr_S
        self.llf = llf
        self.aa = aa
        self.n = n


class CalMultiObj:

    def __init__(self, betas, pre, reside):
        self.betas = betas
        self.pre = pre
        self.reside = reside


class BaseModel:
    """
    Is the parent class of most models
    """

    def __init__(
            self,
            X: Union[np.ndarray, pd.DataFrame, pd.Series],
            y: Union[np.ndarray, pd.DataFrame, pd.Series],
            kernel: str,
            fixed: bool,
            constant: bool,
    ):
        self.X = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        self.y = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
        if len(y.shape) > 1 and y.shape[1] != 1:
            raise ValueError('Label should be one-dimensional arrays')
        if len(y.shape) == 1:
            self.y = self.y.reshape(-1, 1)
        self.kernel = kernel
        self.fixed = fixed
        self.constant = constant
        self.n = X.shape[0]
        if self.constant:
            if len(self.X.shape) == 1 and np.all(self.X == 1):
                raise ValueError("You've already passed in a constant sequence, use constant=False instead")
            for j in range(self.X.shape[1]):
                if np.all(self.X[:, j] == 1):
                    raise ValueError("You've already passed in a constant sequence, use constant=False instead")
            self.X = np.hstack([np.ones((self.n, 1)), X])
        self.k = self.X.shape[1]


class Results(BaseModel):
    """
    Is the result parent class of all models
    """

    def __init__(
            self,
            model,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            kernel: str,
            fixed: bool,
            influ: np.ndarray,
            reside,
            predict_value: np.ndarray,
            betas: np.ndarray,
            tr_STS: float
    ):
        super(Results, self).__init__(X, y, kernel, fixed, constant=False)
        self.model = model
        self.influ = influ
        self.reside = reside
        self.predict_value = predict_value
        self.betas = betas
        self.tr_S = np.sum(influ)
        self.ENP = self.tr_S
        self.tr_STS = tr_STS
        self.TSS = np.sum((y - np.mean(y)) ** 2)
        self.RSS = np.sum(reside ** 2)
        self.sigma2 = self.RSS / (self.n - self.tr_S)
        self.std_res = self.reside / (np.sqrt(self.sigma2 * (1.0 - self.influ)))
        self.cooksD = self.std_res ** 2 * self.influ / (self.tr_S * (1.0 - self.influ))
        self.df_model = self.n - self.tr_S
        self.df_reside = self.n - 2.0 * self.tr_S + self.tr_STS
        self.R2 = 1 - self.RSS / self.TSS
        self.adj_R2 = 1 - (1 - self.R2) * (self.n - 1) / (self.n - self.ENP - 1)
        self.llf = -np.log(self.RSS) * self.n / 2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2
        self.aic = -2.0 * self.llf + 2.0 * (self.tr_S + 1)
        self.aicc = self.aic + 2.0 * self.tr_S * (self.tr_S + 1.0) / (self.n - self.tr_S - 1.0)
        self.bic = -2.0 * self.llf + (self.k + 1) * np.log(self.n)
        self.localR2 = (self.TSS - self.RSS) / self.TSS  # Gaussian核时有效


class GWRResults(Results):

    def __init__(
            self, model, coords, X, y, bw, kernel, fixed, influ, reside, predict_value, betas, CCT, tr_STS
    ):
        """
        betas               : array
                              n*k, estimated coefficients

        predict             : array
                              n*1, predict y values

        CCT                 : array
                              n*k, scaled variance-covariance matrix

        df_model            : integer
                              model degrees of freedom

        df_reside           : integer
                              residual degrees of freedom

        reside              : array
                              n*1, residuals of the response

        RSS                 : scalar
                              residual sum of squares

        CCT                 : array
                              n*k, scaled variance-covariance matrix

        ENP                 : scalar
                              effective number of parameters, which depends on
                              sigma2

        tr_S                : float
                              trace of S (hat) matrix

        tr_STS              : float
                              trace of STS matrix

        R2                  : float
                              R-squared for the entire model (1- RSS/TSS)

        adj_R2              : float
                              adjusted R-squared for the entire model

        aic                 : float
                              Akaike information criterion

        aicc                : float
                              corrected Akaike information criterion
                              to account for model complexity (smaller
                              bandwidths)

        bic                 : float
                              Bayesian information criterion

        sigma2              : float
                              sigma squared (residual variance) that has been
                              corrected to account for the ENP

        std_res             : array
                              n*1, standardised residuals

        bse                 : array
                              n*k, standard errors of parameters (betas)

        influ               : array
                              n*1, leading diagonal of S matrix

        CooksD              : array
                              n*1, Cook's D

        tvalues             : array
                              n*k, local t-statistics

        llf                 : scalar
                              log-likelihood of the full model; see
                              pysal.contrib.glm.family for damily-sepcific
                              log-likelihoods
        """

        super(GWRResults, self).__init__(model,
                                         X, y, kernel, fixed, influ, reside, predict_value, betas, tr_STS)

        self.coords = coords
        self.bw = bw
        self.CCT = CCT * self.sigma2
        self.bse = np.sqrt(self.CCT)
        self.tvalues = self.betas / self.bse
        self.W = np.array([self.model._build_wi(i, self.bw) for i in range(self.n)])

    def adj_alpha(self):
        """
        Corrected alpha (critical) values to account for multiple testing during hypothesis
        testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
        (.01) confidence levels. Correction comes from:

        :cite:`Silva:2016` : da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
        Geographically Weighted Regression. Geographical Analysis.

        """
        alpha = np.array([.1, .05, .001])
        pe = self.ENP
        p = self.k
        return (alpha * p) / pe

    def critical_tval111(self, alpha=None):
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
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals111(self, critical_t=None, alpha=None):
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
        critical        : scalar
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
        n = self.n
        if critical_t is not None:
            critical = np.array(critical_t)
        elif alpha is not None and critical_t is None:
            critical = self.critical_tval(alpha=alpha)
        elif alpha is None and critical_t is None:
            critical = self.critical_tval()

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    def critical_tval(self, alpha=None):
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            adj_alpha = self.adj_alpha()
            alpha = np.abs(adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        if critical_t is not None:
            critical = critical_t
        else:
            critical = self.critical_tval(alpha=alpha)

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues


class GTWRResults(Results):

    def __init__(
            self, model, coords, t, X, y, bw, tau, kernel, fixed, influ, reside, predict_value, betas, CCT, tr_STS
    ):
        """
        tau:        : scalar
                      spatio-temporal scale
        bw_s        : scalar
                      spatial bandwidth
        bw_t        : scalar
                      temporal bandwidth
        See Also GWRResults
        """

        super(GTWRResults, self).__init__(model, X, y, kernel, fixed, influ, reside, predict_value, betas, tr_STS)

        self.coords = coords
        self.t = t
        self.bw = bw
        self.tau = tau
        self.bw_s = self.bw
        self.bw_t = np.sqrt(self.bw ** 2 / self.tau)
        self.CCT = CCT * self.sigma2
        self.bse = np.sqrt(self.CCT)
        self.tvalues = self.betas / self.bse

        # self.W = np.array([self.model._build_wi(i, self.bw, self.tau) for i in range(self.n)])
        try:
            # 尝试创建float32类型的权重矩阵
            self.W = np.array([self.model._build_wi(i, self.bw, self.tau) for i in range(self.n)], dtype=np.float32)
        except MemoryError:
            # 内存不足时的降级方案（可选）
            print("警告：float32存储仍内存不足，不创建权重矩阵...")
            self.W = []
            # self.W = np.zeros((self.n/2, self.n/2), dtype=int32)
            # from scipy import sparse

            # # 构建稀疏矩阵（仅存储非零元素）
            # rows, cols, data = [], [], []
            # for i in range(self.n):
            # wi = self.model._build_wi(i, self.bw, self.tau)
            # non_zero = wi != 0
            # rows.extend([i] * np.sum(non_zero))
            # cols.extend(np.where(non_zero)[0])
            # data.extend(wi[non_zero].astype(np.float32))

            # self.W = sparse.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))

    def adj_alpha(self):
        """
        Corrected alpha (critical) values to account for multiple testing during hypothesis
        testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
        (.01) confidence levels. Correction comes from:

        :cite:`Silva:2016` : da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
        Geographically Weighted Regression. Geographical Analysis.

        """
        alpha = np.array([.1, .05, .001])
        pe = self.ENP
        p = self.k
        return (alpha * p) / pe

    def critical_tval111(self, alpha=None):
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
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals111(self, critical_t=None, alpha=None):
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
        critical        : scalar
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
        n = self.n
        if critical_t is not None:
            critical = np.array(critical_t)
        elif alpha is not None and critical_t is None:
            critical = self.critical_tval(alpha=alpha)
        elif alpha is None and critical_t is None:
            critical = self.critical_tval()

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    def critical_tval(self, alpha=None):
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            adj_alpha = self.adj_alpha()
            alpha = np.abs(adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        if critical_t is not None:
            critical = critical_t
        else:
            critical = self.critical_tval(alpha=alpha)

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues


class MGWRResults(BaseModel):

    def __init__(self, model, coords, X, y, bws, kernel, fixed, bws_history, betas,
                 predict_value, ENP_j, CCT):
        """
        bws         : array-like
                      corresponding spatial bandwidth of all variables
        ENP_j       : array-like
                      effective number of paramters, which depends on
                      sigma2, for each covariate in the model

        See Also GWRResults
        """
        super(MGWRResults, self).__init__(model, X, y, kernel, fixed, constant=False)
        self.coords = coords
        self.bws = bws
        self.bws_history = bws_history
        self.predict_value = predict_value
        self.betas = betas
        self.reside = self.y - self.predict_value
        self.TSS = np.sum((self.y - np.mean(self.y)) ** 2)
        self.RSS = np.sum(self.reside ** 2)
        self.R2 = 1 - self.RSS / self.TSS
        self.llf = -np.log(self.RSS) * self.n / 2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2
        self.bic = -2.0 * self.llf + (self.k + 1) * np.log(self.n)
        if ENP_j is not None:
            self.ENP_j = ENP_j
            self.tr_S = np.sum(self.ENP_j)
            self.ENP = self.tr_S
            self.sigma2 = self.RSS / (self.n - self.tr_S)
            self.CCT = CCT * self.sigma2
            self.bse = np.sqrt(self.CCT)
            self.t_values = self.betas / self.bse
            self.df_model = self.n - self.tr_S
            self.adj_R2 = 1 - (1 - self.R2) * (self.n - 1) / (self.n - self.ENP - 1)
            self.aic = -2.0 * self.llf + 2.0 * (self.tr_S + 1)
            self.aic_c = self.aic + 2.0 * self.tr_S * (self.tr_S + 1.0) / (self.n - self.tr_S - 1.0)

    def tr_S(self):
        return np.sum(self.ENP_j)

    def W(self):
        Ws = []
        for bw_j in self.model.bws:
            W = np.array(
                [self.model._build_wi(i, bw_j) for i in range(self.n)])
            Ws.append(W)
        return Ws

    def adj_alpha_j(self):
        """
        Corrected alpha (critical) values to account for multiple testing during hypothesis
        testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
        (.01) confidence levels. Correction comes from:

        :cite:`Silva:2016` : da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
        Geographically Weighted Regression. Geographical Analysis.

        """
        alpha = np.array([.1, .05, .001])
        pe = np.array(self.ENP_j).reshape((-1, 1))
        p = 1.
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        """
        Utility function to derive the critial t-value based on given alpha
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
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha_j[:, 1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
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
        critical        : scalar
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
        n = self.n
        if critical_t is not None:
            critical = np.array(critical_t)
        elif alpha is not None and critical_t is None:
            critical = self.critical_tval(alpha=alpha)
        elif alpha is None and critical_t is None:
            critical = self.critical_tval()

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    # Function for getting BWs intervals
    def get_bws_intervals(self, selector, level=0.95):
        """
        Computes bandwidths confidence intervals (CIs) for MGWR.
        The CIs are based on Akaike weights and the bandwidth search algorithm used.
        Details are in Li et al. (2020) Annals of AAG

        Returns a list of confidence intervals. e.g. [(40, 60), (100, 180), (150, 300)]

        """
        intervals = []
        try:
            import pandas as pd
        except ImportError:
            return

        for j in range(self.k):
            # Get AICcs and associated bw from the last iteration of back-fitting and make a DataFrame
            aiccs = pd.DataFrame(list(zip(*selector.sel_hist[-self.k + j]))[1], columns=["aicc"])
            aiccs['bw'] = list(zip(*selector.sel_hist[-self.k + j]))[0]
            # Sort DataFrame by the AICc values
            aiccs = aiccs.sort_values(by=['aicc'])
            # Calculate delta AICc
            d_aic_ak = aiccs.aicc - aiccs.aicc.min()
            # Calculate AICc weights
            w_aic_ak = np.exp(-0.5 * d_aic_ak) / np.sum(np.exp(-0.5 * d_aic_ak))
            aiccs['w_aic_ak'] = w_aic_ak / np.sum(w_aic_ak)
            # Calculate cum. AICc weights
            aiccs['cum_w_ak'] = aiccs.w_aic_ak.cumsum()
            # Find index where the cum weights above p-val
            index = len(aiccs[aiccs.cum_w_ak < level]) + 1
            # Get bw boundaries
            interval = (aiccs.iloc[:index, :].bw.min(), aiccs.iloc[:index, :].bw.max())
            intervals += [interval]
        return intervals

    def local_collinearity(self):
        """
        Computes several indicators of multicollinearity within a geographically
        weighted design matrix, including:

        local condition number (n, 1)
        local variance-decomposition proportions (n, p)

        Returns four arrays with the order and dimensions listed above where n
        is the number of locations used as calibrations points and p is the
        nubmer of explanatory variables

        """
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = self.n
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))

        for i in range(nrow):
            xw = np.zeros((x.shape))
            for j in range(nvar):
                wi = w[j][i]
                sw = np.sum(wi)
                wi = wi / sw
                xw[:, j] = x[:, j] * wi

            sxw = np.sqrt(np.sum(xw ** 2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i,] = svdx[1][0] / svdx[1]

            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi ** 2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return local_CN, VDP

    # 未测试
    def spatial_variability(self, selector, n_iters=1000, seed=None):
        """
        Method to compute a Monte Carlo test of spatial variability for each
        estimated coefficient surface.

        WARNING: This test is very computationally demanding!

        Parameters
        ----------
        selector        : sel_bw object
                          should be the sel_bw object used to select a bandwidth
                          for the gwr model that produced the surfaces that are
                          being tested for spatial variation

        n_iters         : int
                          the number of Monte Carlo iterations to include for
                          the tests of spatial variability.

        seed            : int
                          optional parameter to select a custom seed to ensure
                          stochastic results are replicable. Default is none
                          which automatically sets the seed to 5536

        Returns
        -------

        p values        : list
                          a list of psuedo p-values that correspond to the model
                          parameter surfaces. Allows us to assess the
                          probability of obtaining the observed spatial
                          variation of a given surface by random chance.


        """
        temp_sel = copy.deepcopy(selector)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        search_params = temp_sel.search_params

        if self.model.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        try:
            from tqdm.auto import tqdm  # if they have it, let users have a progress bar
        except ImportError:

            def tqdm(x, desc=''):  # otherwise, just passthrough the range
                return x

        for x in tqdm(range(n_iters), desc='Testing'):
            temp_coords = np.random.permutation(self.model.coords)
            temp_sel.coords = temp_coords
            temp_sel.search(**search_params)
            temp_params = temp_sel.params
            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals


class MGTWRResults(MGWRResults):

    def __init__(self, model, coords, t, X, y, bws, taus, kernel, fixed, bw_ts, bws_history, taus_history, betas,
                 predict_value, ENP_j, CCT):
        """
        taus        : array-like
                     corresponding spatio-temporal scale of all variables
        bws         : array-like
                     corresponding spatio bandwidth of all variables
        bw_ts       : array-like
                     corresponding temporal bandwidth of all variables
        See Also
        -------------
        MGWRResults
        GWRResults
        """
        super(MGTWRResults, self).__init__(model,
                                           coords, X, y, bws, kernel, fixed, bws_history, betas, predict_value, ENP_j,
                                           CCT)
        self.t = t
        self.taus = taus
        self.bw_ts = bw_ts
        self.taus_history = taus_history

    def adj_alpha_j(self):
        """
        Corrected alpha (critical) values to account for multiple testing during hypothesis
        testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
        (.01) confidence levels. Correction comes from:

        :cite:`Silva:2016` : da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
        Geographically Weighted Regression. Geographical Analysis.

        """
        alpha = np.array([.1, .05, .001])
        pe = np.array(self.ENP_j).reshape((-1, 1))
        p = 1.
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        """
        Utility function to derive the critial t-value based on given alpha
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
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha_j[:, 1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
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
        critical        : scalar
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
        n = self.n
        if critical_t is not None:
            critical = np.array(critical_t)
        elif alpha is not None and critical_t is None:
            critical = self.critical_tval(alpha=alpha)
        elif alpha is None and critical_t is None:
            critical = self.critical_tval()

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues
