"""
Classes for full conformal prediction for exchangeable data and data under standard and feedback covariate shift,
both for black-box predictive models and computationally optimized for ridge regression.
Throughout this file, variable name suffixes denote the shape of the numpy array, where
    n: number of training points, or generic number of data points
    n1: n + 1
    p: number of features
    y: number of candidate labels, |Y|
    u: number of sequences in domain, |X|
    m: number of held-out calibration points for split conformal methods
"""

import numpy as np
import time
import scipy as sc

from abc import ABC, abstractmethod

# ===== utilities for split conformal =====

def get_split_coverage(lu_nx2, y_n):
    """
    Computes empirical coverage of split conformal confidence interval
    :param lu_nx2: (n, 2) numpy array where first and second columns are lower and upper endpoints
    :param y_n: (n,) numpy array of true labels
    :return: float, empirical coverage
    """
    cov = np.sum((y_n >= lu_nx2[:, 0]) & (y_n <= lu_nx2[:, 1])) / y_n.size
    return cov

def get_randomized_staircase_coverage(C_n, y_n):
    """
    Computes empirical coverage and lengths of randomized staircase confidence sets.

    :param C_n: length-n list of outputs of get_randomized_staircase_confidence_set (i.e., list of tuples)
    :param y_n: (n,) numpy array of true labels
    :return: (n,) binary array of coverage and (n,) numpy array of lengths
    """
    def is_covered(confint_list, y):
        for confint_2 in confint_list:
            if y >= confint_2[0] and y <= confint_2[1]:
                return True
        return False
    def get_len_conf_set(confint_list):
        return np.sum([confint_2[1] - confint_2[0] for confint_2 in confint_list])

    cov_n = np.array([is_covered(confset, y) for confset, y in zip(C_n, y_n)])
    len_n = np.array([get_len_conf_set(confset) for confset in C_n])
    return cov_n, len_n

def get_randomized_staircase_confidence_set(scores_m, weights_m1, predtest, alpha: float = 0.1):
    """
    Computes the "randomized staircase" confidence set in Alg. S1.

    :param scores_m: (m,) numpy array of calibration scores
    :param weights_m1: (m + 1) numpy array of calibration weights and single test weight
    :param predtest: float, prediction on test input
    :param alpha: miscoverage level
    :return: list of tuples (l, u), where l and u are floats denoting lower and upper
        endpoints of an interval.
    """
    lb_is_set = False
    idx = np.argsort(scores_m)
    sortedscores_m1 = np.hstack([0, scores_m[idx]])
    sortedweights_m1 = np.hstack([0, weights_m1[: -1][idx]])
    C = []

    # interval that is deterministically included in the confidence set
    # (color-coded green in Fig. S1)
    cdf_m1 = np.cumsum(sortedweights_m1) # CDF up to i-th sorted calibration score
    cdf_plus_test_weight_m1 = cdf_m1 + weights_m1[-1]
    deterministic_idx = np.where(cdf_plus_test_weight_m1 < 1 - alpha)[0]
    if deterministic_idx.size:
        i_det = np.max(deterministic_idx)
        C.append((predtest - sortedscores_m1[i_det + 1], predtest + sortedscores_m1[i_det + 1]))

    # intervals that are randomly included in the confidence set
    # (color-coded teal and blue in Fig. S1)
    for i in range(i_det + 1, sortedscores_m1.size - 1):
        assert(cdf_plus_test_weight_m1[i] >= 1 - alpha)
        if cdf_plus_test_weight_m1[i] >= 1 - alpha and cdf_m1[i] < 1 - alpha:
            if not lb_is_set:
                lb_is_set = True
                LF = cdf_m1[i]
            F = (cdf_plus_test_weight_m1[i] - (1 - alpha)) / (cdf_m1[i] + weights_m1[-1] - LF)
            if sc.stats.bernoulli.rvs(1 - F):
                C.append((predtest + sortedscores_m1[i], predtest + sortedscores_m1[i + 1]))
                C.append((predtest - sortedscores_m1[i + 1], predtest - sortedscores_m1[i]))

    # halfspaces that are randomly included in the confidence set
    # (color-coded purple in Fig. S1)
    if cdf_m1[-1] < 1 - alpha:  # sum of all calibration weights
        if not lb_is_set:
            LF = cdf_m1[-1]
        F = alpha / (1 - LF)
        if sc.stats.bernoulli.rvs(1 - F):
            C.append((predtest + sortedscores_m1[-1], np.inf))
            C.append((-np.inf, predtest - sortedscores_m1[-1]))
    return C



# ========== full conformal utilities ==========

def get_weighted_quantile(quantile, w_n1xy, scores_n1xy):
    """
    Compute the quantile of weighted scores for each candidate label y

    :param quantile: float, quantile
    :param w_n1xy: (n + 1, |Y|) numpy array of weights (unnormalized)
    :param scores_n1xy: (n + 1, |Y|) numpy array of scores
    :return: (|Y|,) numpy array of quantiles
    """
    if w_n1xy.ndim == 1:
        w_n1xy = w_n1xy[:, None]
        scores_n1xy = scores_n1xy[:, None]

    # normalize probabilities
    p_n1xy = w_n1xy / np.sum(w_n1xy, axis=0)

    # sort scores and their weights accordingly
    sorter_per_y_n1xy = np.argsort(scores_n1xy, axis=0)
    sortedscores_n1xy = np.take_along_axis(scores_n1xy, sorter_per_y_n1xy, axis=0)
    sortedp_n1xy = np.take_along_axis(p_n1xy, sorter_per_y_n1xy, axis=0)

    # locate quantiles of weighted scores per y
    cdf_n1xy = np.cumsum(sortedp_n1xy, axis=0)
    qidx_y = np.sum(cdf_n1xy < quantile, axis=0)  # equivalent to [np.searchsorted(cdf_n1, q) for cdf_n1 in cdf_n1xy]
    q_y = sortedscores_n1xy[(qidx_y, range(qidx_y.size))]
    return q_y

def is_covered(y, confset, y_increment):
    """
    Return if confidence set covers true label

    :param y: true label
    :param confset: numpy array of values in confidence set
    :param y_increment: float, \Delta increment between candidate label values, 0.01 in main paper
    :return: bool
    """
    return np.any(np.abs(y - confset) < (y_increment / 2))



# ========== utilities and classes for full conformal with ridge regression ==========

def get_invcov_dot_xt(X_nxp, gamma, use_lapack: bool = True):
    """
    Compute (X^TX + \gamma I)^{-1} X^T

    :param X_nxp: (n, p) numpy array encoding sequences
    :param gamma: float, ridge regularization strength
    :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
    :return: (p, n) numpy array, (X^TX + \gamma I)^{-1} X^T
    """
    reg_pxp = gamma * np.eye(X_nxp.shape[1])
    reg_pxp[0, 0] = 0  # don't penalize intercept term
    cov_pxp = X_nxp.T.dot(X_nxp) + reg_pxp
    if use_lapack:
        # fastest way to invert PD matrices from
        # https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
        zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
        invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
        assert(info == 0)
        invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
    else:
        invcov_pxp = sc.linalg.pinvh(cov_pxp)
    return invcov_pxp.dot(X_nxp.T)


class ConformalRidge(ABC):
    """
    Abstract base class for full conformal with computations optimized for ridge regression.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        """
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        :param gamma: float, ridge regularization strength
        :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
        """
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
        self.ys = ys
        self.n_y = ys.size
        self.gamma = gamma
        self.use_lapack = use_lapack

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    def get_insample_scores(self, Xaug_n1xp, ytrain_n):
        """
        Compute in-sample scores, i.e. residuals using model trained on all n + 1 data points (instead of LOO data)

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :return: (n + 1, |Y|) numpy array of scores
        """
        A = get_invcov_dot_xt(Xaug_n1xp, self.gamma, use_lapack=self.use_lapack)
        C = A[:, : -1].dot(ytrain_n)  # p elements
        a_n1 = C.dot(Xaug_n1xp.T)
        b_n1 = A[:, -1].dot(Xaug_n1xp.T)

        # process in-sample scores for each candidate value y
        scoresis_n1xy = np.zeros([ytrain_n.size + 1, self.n_y])
        by_n1xy = np.outer(b_n1, self.ys)
        muhatiy_n1xy = a_n1[:, None] + by_n1xy
        scoresis_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_n1xy[: -1])
        scoresis_n1xy[-1] = np.abs(self.ys - muhatiy_n1xy[-1])
        return scoresis_n1xy

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, compute_lrs: bool = True):
        """
        Compute LOO scores, i.e. residuals using model trained on n data points (training + candidate test points,
        but leave i-th training point out).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
        :param compute_lrs: bool: whether or not to compute likelihood ratios (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        # fit n + 1 LOO models and store linear parameterizations of \mu_{-i, y}(X_i) as function of y
        n = ytrain_n.size
        ab_nx2 = np.zeros([n, 2])
        C_nxp = np.zeros([n, self.p])
        An_nxp = np.zeros([n, self.p])
        for i in range(n):
            # construct A_{-i}
            Xi_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]]) # n rows
            Ai = get_invcov_dot_xt(Xi_nxp, self.gamma, use_lapack=self.use_lapack)

            # compute linear parameterizations of \mu_{-i, y}(X_i)
            yi_ = np.hstack([ytrain_n[: i], ytrain_n[i + 1 :]])  # n - 1 elements
            Ci = Ai[:, : -1].dot(yi_) # p elements
            ai = Ci.dot(Xaug_n1xp[i])  # = Xtrain_nxp[i]
            bi = Ai[:, -1].dot(Xaug_n1xp[i])

            # store
            ab_nx2[i] = ai, bi
            C_nxp[i] = Ci
            An_nxp[i] = Ai[:, -1]

        # LOO score for i = n + 1
        tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
        beta_p = tmp.dot(ytrain_n)
        alast = beta_p.dot(Xaug_n1xp[-1])  # prediction a_{n + 1}. Xaug_n1xp[-1] = Xtest_p

        # process LOO scores for each candidate value y
        scoresloo_n1xy = np.zeros([n + 1, self.n_y])
        by_nxy = np.outer(ab_nx2[:, 1], self.ys)
        prediy_nxy = ab_nx2[:, 0][:, None] + by_nxy
        scoresloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - prediy_nxy)
        scoresloo_n1xy[-1] = np.abs(self.ys - alast)

        # likelihood ratios for each candidate value y
        w_n1xy = None
        if compute_lrs:
            betaiy_nxpxy = C_nxp[:, :, None] + self.ys * An_nxp[:, :, None]
            # compute normalizing constant in Eq. 6 in main paper
            pred_nxyxu = np.tensordot(betaiy_nxpxy, self.Xuniv_uxp, axes=(1, 1))
            normconst_nxy = np.sum(np.exp(lmbda * pred_nxyxu), axis=2)
            ptrain_n = self.ptrain_fn(Xaug_n1xp[: -1])

            w_n1xy = np.zeros([n + 1, self.n_y])
            wi_num_nxy = np.exp(lmbda * prediy_nxy)
            w_n1xy[: -1] = wi_num_nxy / (ptrain_n[:, None] * normconst_nxy)

            # for last i = n + 1, which is constant across candidate values of y
            Z = self.get_normalizing_constant(beta_p, lmbda)
            w_n1xy[-1] = np.exp(lmbda * alast) / (self.ptrain_fn(Xaug_n1xp[-1][None, :]) * Z)
        return scoresloo_n1xy, w_n1xy

    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1, use_is_scores: bool = False):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

        # ===== compute scores and weights =====

        # compute in-sample scores
        scoresis_n1xy = self.get_insample_scores(Xaug_n1xp, ytrain_n) if use_is_scores else None

        # compute LOO scores and likelihood ratios
        scoresloo_n1xy, w_n1xy = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # ===== construct confidence sets =====

        # based on LOO score
        looq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresloo_n1xy)
        loo_cs = self.ys[scoresloo_n1xy[-1] <= looq_y]

        # based on in-sample score
        is_cs = None
        if use_is_scores:
            isq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresis_n1xy)
            is_cs = self.ys[scoresis_n1xy[-1] <= isq_y]
        return loo_cs, is_cs


class ConformalRidgeExchangeable(ConformalRidge):
    """
    Class for full conformal with ridge regression, assuming exchangeable data.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        scoresloo_n1xy, _ = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=False)
        # for exchangeble data, equal weights on all data points (no need to compute likelihood ratios in line above)
        w_n1xy = np.ones([Xaug_n1xp.shape[0], self.n_y])
        return scoresloo_n1xy, w_n1xy


class ConformalRidgeFeedbackCovariateShift(ConformalRidge):
    """
    Class for full conformal with ridge regression under feedback covariate shift via Eq. 6 in main paper.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        scoresloo_n1xy, w_n1xy = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=True)
        return scoresloo_n1xy, w_n1xy


class ConformalRidgeStandardCovariateShift(ConformalRidge):
    """
    Class for full conformal with ridge regression under standard covariate shift.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        # fit model to training data
        tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
        beta_p = tmp.dot(ytrain_n)

        # compute normalizing constant for test covariate distribution
        Z = self.get_normalizing_constant(beta_p, lmbda)

        # get likelihood ratios for n + 1 covariates
        pred_n1 = Xaug_n1xp.dot(beta_p)
        ptest_n1 = np.exp(lmbda * pred_n1) / Z
        w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
        return w_n1

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        # LOO scores
        scoresloo_n1xy, _ = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=False)

        # compute likelihood ratios
        w_n1 = self.get_lrs(Xaug_n1xp, ytrain_n, lmbda)
        w_n1xy = w_n1[:, None] * np.ones([Xaug_n1xp.shape[0], self.n_y])
        return scoresloo_n1xy, w_n1xy



# ========== utilities and classes for full conformal with black-box model ==========

def get_scores(model, Xaug_nxp, yaug_n, use_loo_score: bool = False):
    if use_loo_score:
        n1 = yaug_n.size  # n + 1
        scores_n1 = np.zeros([n1])

        for i in range(n1):
            Xtrain_nxp = np.vstack([Xaug_nxp[: i], Xaug_nxp[i + 1 :]])
            ytrain_n = np.hstack([yaug_n[: i], yaug_n[i + 1 :]])

            # train on LOO dataset
            model.fit(Xtrain_nxp, ytrain_n)
            pred_1 = model.predict(Xaug_nxp[i][None, :])
            scores_n1[i] = np.abs(yaug_n[i] - pred_1[0])

    else:  # in-sample score
        model.fit(Xaug_nxp, yaug_n)
        pred_n1 = model.predict(Xaug_nxp)
        scores_n1 = np.abs(yaug_n - pred_n1)
    return scores_n1


class Conformal(ABC):
    """
    Abstract base class for full conformal with black-box predictive model.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        """
        :param model: object with predict() method
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: (|Y|,) numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        """
        self.model = model
        self.ptrain_fn = ptrain_fn
        self.ys = ys
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
        self.n_y = ys.size

    @abstractmethod
    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda,
                           use_loo_score: bool = True, alpha: float = 0.1, print_every: int = 10, verbose: bool = True):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))

        np.set_printoptions(precision=3)
        cs, n = [], ytrain_n.size
        t0 = time.time()
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        scores_n1xy = np.zeros([n + 1, self.n_y])
        w_n1xy = np.zeros([n + 1, self.n_y])

        for y_idx, y in enumerate(self.ys):

            # get scores
            yaug_n1 = np.hstack([ytrain_n, y])
            scores_n1 = get_scores(self.model, Xaug_n1xp, yaug_n1, use_loo_score=use_loo_score)
            scores_n1xy[:, y_idx] = scores_n1

            # get likelihood ratios
            w_n1 = self.get_lrs(Xaug_n1xp, yaug_n1, lmbda)
            w_n1xy[:, y_idx] = w_n1

            # for each value of inverse temperature lambda, compute quantile of weighted scores
            q = get_weighted_quantile(1 - alpha, w_n1, scores_n1)

            # if y <= quantile, include in confidence set
            if scores_n1[-1] <= q:
                cs.append(y)

            # print progress
            if verbose and (y_idx + 1) % print_every == 0:
                print("Done with {} / {} y values ({:.1f} s)".format(
                    y_idx + 1, self.ys.size, time.time() - t0))
        return np.array(cs), scores_n1xy, w_n1xy


class ConformalExchangeable(Conformal):
    """
    Full conformal with black-box predictive model, assuming exchangeable data.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        return np.ones([Xaug_n1xp.shape[0]])


class ConformalFeedbackCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under feedback covariate shift via Eq. 6 in main paper.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # compute weights for each value of lambda, the inverse temperature
        w_n1 = np.zeros([yaug_n1.size])
        for i in range(yaug_n1.size):

            # fit LOO model
            Xtr_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]])
            ytr_n = np.hstack([yaug_n1[: i], yaug_n1[i + 1 :]])
            self.model.fit(Xtr_nxp, ytr_n)

            # compute normalizing constant
            predall_n = self.model.predict(self.Xuniv_uxp)
            Z = np.sum(np.exp(lmbda * predall_n))

            # compute likelihood ratios
            testpred = self.model.predict(Xaug_n1xp[i][None, :])
            ptest = np.exp(lmbda * testpred) / Z
            w_n1[i] = ptest / self.ptrain_fn(Xaug_n1xp[i][None, :])
        return w_n1


class ConformalStandardCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under standard covariate shift.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # get normalization constant for test covariate distribution
        self.model.fit(Xaug_n1xp[: -1], yaug_n1[: -1])  # Xtrain_nxp, ytrain_n
        predall_u = self.model.predict(self.Xuniv_uxp)
        Z = np.sum(np.exp(lmbda * predall_u))

        # get likelihood ratios
        pred_n1 = self.model.predict(Xaug_n1xp)
        ptest_n1 = np.exp(lmbda * pred_n1) / Z
        w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
        return w_n1
