import numpy as np
from spglm.family import Gaussian, Binomial, Poisson
from spglm.glm import GLM
from mgtwr.diagnosis import get_AICc, get_AIC, get_BIC, get_CV


def summaryModel(Result):
    summary = '=' * 75 + '\n'
    summary += "%-54s %20s\n" % ('Model type', Result.kernel)
    summary += "%-60s %14d\n" % ('Number of observations:', Result.n)
    summary += "%-60s %14d\n\n" % ('Number of covariates:', Result.k)
    return summary


def summaryGLM(Result):
    XNames = ["X" + str(i) for i in range(Result.k)]
    # glm_rslt = GLM(Result.model.y, Result.model.X, constant=False,
    #               family=Result.family).fit()
    glm_rslt = GLM(Result.y, Result.X, constant=False).fit()

    summary = "%s\n" % ('Global Regression Results')
    summary += '-' * 75 + '\n'

    if Result.kernel == 'gaussian':
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', glm_rslt.deviance)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', glm_rslt.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', glm_rslt.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', get_AICc(glm_rslt))
        summary += "%-62s %12.3f\n" % ('BIC:', glm_rslt.bic)
        summary += "%-62s %12.3f\n" % ('R2:', glm_rslt.D2)
        summary += "%-62s %12.3f\n\n" % ('Adj. R2:', glm_rslt.adj_D2)
    else:
        summary += "%-62s %12.3f\n" % ('Deviance:', glm_rslt.deviance)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', glm_rslt.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', glm_rslt.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', get_AICc(glm_rslt))
        summary += "%-62s %12.3f\n" % ('BIC:', glm_rslt.bic)
        summary += "%-62s %12.3f\n" % ('Percent deviance explained:', glm_rslt.D2)
        summary += "%-62s %12.3f\n\n" % ('Adj. percent deviance explained:', glm_rslt.adj_D2)

    summary += "%-31s %10s %10s %10s %10s\n" % ('Variable', 'Est.', 'SE', 't(Est/SE)', 'p-value')
    summary += "%-31s %10s %10s %10s %10s\n" % ('-' * 31, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    for i in range(Result.k):
        summary += "%-31s %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i][:30], glm_rslt.params[i], glm_rslt.bse[i],
            glm_rslt.tvalues[i], glm_rslt.pvalues[i])
    summary += "\n"
    return summary


def summaryGWR(Result):
    XNames = ["X" + str(i) for i in range(Result.k)]

    summary = "%s\n" % ('Geographically Weighted Regression (GWR) Results')
    summary += '-' * 75 + '\n'

    if Result.fixed == True:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + Result.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + Result.kernel)

    summary += "%-62s %12.3f\n" % ('Bandwidth used:', Result.bw)

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'

    if Result.kernel == 'gaussian':
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', Result.RSS)
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Result.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Result.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(Result.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Result.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Result.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Result.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Result.bic)
        summary += "%-62s %12.3f\n" % ('R2:', Result.R2)
        summary += "%-62s %12.3f\n" % ('Adjusted R2:', Result.adj_R2)

    else:
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Result.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Result.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Result.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Result.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Result.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Result.bic)
        summary += "%-62s %12.3f\n" % ('R2:', Result.R2)
        summary += "%-62s %12.3f\n" % ('Adjusted R2:', Result.adj_R2)
    #     summary += "%-60s %12.3f\n" % ('Percent deviance explained:', Result.D2)
    #     summary += "%-60s %12.3f\n" % ('Adjusted percent deviance explained:', Result.adj_D2)
    #
    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', Result.adj_alpha()[1])
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):', Result.critical_tval(Result.adj_alpha())[1])

    summary += "\n%s\n" % ('Summary Statistics For GWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD', 'Min', 'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    for i in range(Result.k):  # Result.betas估计系数
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i][:20], np.mean(Result.betas[:, i]), np.std(Result.betas[:, i]),
            np.min(Result.betas[:, i]), np.median(Result.betas[:, i]),
            np.max(Result.betas[:, i]))

    summary += '=' * 75 + '\n'

    return summary


def summaryGTWR(Result):
    XNames = ["X" + str(i) for i in range(Result.k)]

    summary = ''
    summary += "%s\n" % ('Geographically and temporally weighted regression (GTWR) Results')
    summary += '-' * 75 + '\n'

    if Result.fixed == True:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + Result.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + Result.kernel)

    summary += "%-54s %20s\n" % ('Criterion for optimal bandwidth:', Result.bw)
    summary += "%-62s %12.3f\n" % ('spatio-temporal scale used:', Result.tau)

    # if Result.model.selector.rss_score:
    #     summary += "%-54s %20s\n" % ('Score of Change (SOC) type:', 'RSS')
    # else:
    #     summary += "%-54s %20s\n" % ('Score of Change (SOC) type:', 'Smoothing f')
    #
    # summary += "%-54s %20s\n\n" % ('Termination criterion for GTWR:', Result.model.selector.tol_multi)

    # summary += "%s\n" % ('MGWR bandwidths')
    # summary += '-' * 75 + '\n'
    # summary += "%-15s %14s %10s %16s %16s\n" % ('Variable', 'Bandwidth', 'ENP_j', 'Adj t-val(95%)', 'Adj alpha(95%)')
    # for j in range(Result.k):
    #     summary += "%-14s %15.3f %10.3f %16.3f %16.3f\n" % (
    #         XNames[j], Result.model.bw[j], Result.ENP_j[j],
    #         Result.critical_tval()[j], Result.adj_alpha_j()[j])

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'

    if Result.kernel == 'gaussian':
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', Result.RSS)
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Result.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Result.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(Result.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Result.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Result.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Result.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Result.bic)
        summary += "%-62s %12.3f\n" % ('R2:', Result.R2)
        summary += "%-62s %12.3f\n" % ('Adjusted R2:', Result.adj_R2)

    else:
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Result.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Result.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', Result.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', Result.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', Result.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', Result.bic)
        summary += "%-62s %12.3f\n" % ('R2:', Result.R2)
        summary += "%-62s %12.3f\n" % ('Adjusted R2:', Result.adj_R2)
    #     summary += "%-60s %12.3f\n" % ('Percent deviance explained:', Result.D2)
    #     summary += "%-60s %12.3f\n" % ('Adjusted percent deviance explained:', Result.adj_D2)
    #
    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', Result.adj_alpha()[1])
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):', Result.critical_tval(Result.adj_alpha())[1])

    summary += "\n%s\n" % ('Summary Statistics For GTWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD', 'Min', 'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    for i in range(Result.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i][:20], np.mean(Result.betas[:, i]), np.std(Result.betas[:, i]),
            np.min(Result.betas[:, i]), np.median(Result.betas[:, i]),
            np.max(Result.betas[:, i]))

    summary += '=' * 75 + '\n'

    return summary


def summaryMGWR(Result):
    XNames = ["X" + str(i) for i in range(Result.k)]

    summary = ''
    summary += "%s\n" % ('Multi-Scale Geographically Weighted Regression (MGWR) Results')
    summary += '-' * 75 + '\n'

    if Result.fixed == True:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + Result.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + Result.kernel)

    summary += "%-54s %20s\n" % ('Criterion for optimal bandwidth:', Result.model.selector.criterion)

    if Result.model.selector.rss_score:
        summary += "%-54s %20s\n" % ('Score of Change (SOC) type:', 'RSS')
    else:
        summary += "%-54s %20s\n" % ('Score of Change (SOC) type:', 'Smoothing f')

    summary += "%-54s %20s\n\n" % ('Termination criterion for MGWR:', Result.model.selector.tol_multi)

    summary += "%s\n" % ('MGWR bandwidths')
    summary += '-' * 75 + '\n'
    summary += "%-15s %14s %10s %16s %16s\n" % ('Variable', 'Bandwidth', 'ENP_j', 'Adj t-val(95%)', 'Adj alpha(95%)')
    for j in range(Result.k):
        summary += "%-14s %15.3f %10.3f %16.3f %16.3f\n" % (
            XNames[j], Result.model.bws[j], Result.ENP_j[j],
            Result.critical_tval()[j], Result.adj_alpha_j()[j])

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'

    summary += "%-62s %12.3f\n" % ('Residual sum of squares:', Result.resid_ss)
    summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', Result.tr_S)
    summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', Result.df_model)

    summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(Result.sigma2))
    summary += "%-62s %12.3f\n" % ('Log-likelihood:', Result.llf)
    summary += "%-62s %12.3f\n" % ('AIC:', Result.aic)
    summary += "%-62s %12.3f\n" % ('AICc:', Result.aicc)
    summary += "%-62s %12.3f\n" % ('BIC:', Result.bic)
    summary += "%-62s %12.3f\n" % ('R2', Result.R2)
    summary += "%-62s %12.3f\n" % ('Adjusted R2', Result.adj_R2)

    summary += "\n%s\n" % ('Summary Statistics For MGWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD', 'Min', 'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
    for i in range(Result.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
            XNames[i][:20], np.mean(Result.params[:, i]), np.std(Result.params[:, i]),
            np.min(Result.params[:, i]), np.median(Result.params[:, i]),
            np.max(Result.params[:, i]))
    summary += '=' * 75 + '\n'
    return summary
