import numpy as np
import time
import scipy as sc

from abc import ABC, abstractmethod

# ========== conformal utilities ==========

def get_wt_centered_train_data(wt_is_1, n_sample, p_mutate, seed: int = None):
    np.random.seed(seed)
    X_nxp = sc.stats.bernoulli.rvs(1 - p_mutate if wt_is_1 else p_mutate, size=(n_sample, 13))
    X_nxp[X_nxp == 0] = -1

    def ptrain_fn(Xtest_nxp):
        nzero_n = np.sum(Xtest_nxp[:, 1 : 14] < 0, axis=1)
        none_n = 13 - nzero_n
        pmut_n = np.power(p_mutate, nzero_n if wt_is_1 else none_n)
        pwt_n = np.power(1 - p_mutate, none_n if wt_is_1 else nzero_n)
        return pmut_n * pwt_n

    return X_nxp, ptrain_fn

def weighted_quantile(vals_n, weights_n, quantile, tol: float = 1e-12):
    if np.abs(np.sum(weights_n) - 1) > tol:
        raise ValueError("weights don't sum to one.")

    # sort values
    sorter = np.argsort(vals_n)
    vals_n = vals_n[sorter]
    weights_n = weights_n[sorter]

    cumweight_n = np.cumsum(weights_n)
    idx = np.searchsorted(cumweight_n, quantile, side='left')
    return vals_n[idx]

def get_quantile(alpha, w_n1xy, scores_n1xy):
    """
    Returns weighted 1 - alpha quantile of n + 1 scores, given likelihood ratios w(x_i; z_{-i}) and scores
    :param alpha: float, miscoverage level
    :param w_n1: numpy array, n + 1 likelihood ratios, assuming the last one is w(x_{n + 1}; z_{1:n})
    :param scores_n1: numpy array, n + 1 scores
    :return: float
    """
    if w_n1xy.ndim == 1:
        w_n1xy = w_n1xy[:, None]
        scores_n1xy = scores_n1xy[:, None]
    p_n1xy = w_n1xy / np.sum(w_n1xy, axis=0)
    augscore_n1xy = np.vstack([scores_n1xy[: -1], np.inf * np.ones([scores_n1xy.shape[1]])])
    q_y = np.array([weighted_quantile(augscore_n1, p_n1, 1 - alpha) for augscore_n1, p_n1 in zip(augscore_n1xy.T, p_n1xy.T)])
    return q_y

def is_covered(y, cs, tol):
    return np.any(np.abs(y - cs) < tol)




# ========== ridge regression calibration ==========

def get_invcov_dot_xt(X_nxp, gamma, use_lapack: bool = True):
    reg_pxp = gamma * np.eye(X_nxp.shape[1])
    reg_pxp[0, 0] = 0  # don't penalize intercept term
    cov_pxp = X_nxp.T.dot(X_nxp) + reg_pxp
    if use_lapack:
        # fastest way to invert PD matrices, but no robust error-checking
        # https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
        zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
        invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
        assert(info == 0)
        invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
    else:
        invcov_pxp = sc.linalg.pinvh(cov_pxp)
    return invcov_pxp.dot(X_nxp.T)


class ConformalRidge(ABC):
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
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

        w_n1xy = None
        # likelihood ratios for each candidate value y
        if compute_lrs:
            betaiy_nxpxy = C_nxp[:, :, None] + self.ys * An_nxp[:, :, None]
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

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

        # ===== compute scores and weights =====

        # compute in-sample scores
        scoresis_n1xy = self.get_insample_scores(Xaug_n1xp, ytrain_n)

        # compute LOO scores and likelihood ratios
        scoresloo_n1xy, w_n1xy = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # ===== construct confidence sets =====

        # based on LOO score
        looq_y = get_quantile(alpha, w_n1xy, scoresloo_n1xy)
        loo_cs = self.ys[scoresloo_n1xy[-1] <= looq_y]

        # based on in-sample score
        isq_y = get_quantile(alpha, w_n1xy, scoresis_n1xy)
        is_cs = self.ys[scoresis_n1xy[-1] <= isq_y]
        return loo_cs, is_cs


class ConformalRidgeCovariateIntervention(ConformalRidge):
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        scoresloo_n1xy, w_n1xy = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=True)
        return scoresloo_n1xy, w_n1xy



class ConformalRidgeCovariateShift(ConformalRidge):
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



# ========== model-agnostic calibration ==========

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
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
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
                           use_loo_score: bool = True, alpha: float = 0.1,
                           print_every: int = 10, verbose: bool = True):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))

        np.set_printoptions(precision=3)
        cs, n = [], ytrain_n.size
        t0 = time.time()
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        scores_n1xy = np.zeros([n + 1, self.n_y]),
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
            q = get_quantile(alpha, w_n1, scores_n1)

            # if y <= quantile, include in confidence set
            if scores_n1[-1] <= q:
                cs.append(y)

            # print progress
            if verbose and (y_idx + 1) % print_every == 0:
                print("Done with {} / {} y values ({:.1f} s): {}".format(
                    y_idx + 1, self.ys.size, time.time() - t0, np.array(cs)))
        return np.array(cs), scores_n1xy, w_n1xy



class ConformalCovariateIntervention(Conformal):
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


class ConformalCovariateShift(Conformal):
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
