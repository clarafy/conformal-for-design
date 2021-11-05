import numpy as np
import time
import scipy as sc

from sklearn.linear_model import Ridge



def get_lrs(Xaug_nxp, yaug_n, ptrain_fn, Xuniv_nxp, lmbda, model):
    """
    Returns n + 1 likelihood ratios w(x_i; z_{-i}) for i = 1, ..., n + 1
    :param Xaug_nxp: numpy array of n + 1 covariates (training + test)
    :param yaug_n: numpy array of n + 1 labels (training + test)
    :param ptrain_fn: fn returning likelihood of covariate under training distribution
    :param model: fit() and predict() methods
    :return: numpy array l x (n + 1) likelihood ratios where l is number of lambda values
    """

    # compute weights for each value of lambda, the inverse temperature
    w_n1 = np.zeros([yaug_n.size])
    for i in range(yaug_n.size):

        # fit LOO model
        Xtr_nxp = np.vstack([Xaug_nxp[: i], Xaug_nxp[i + 1 :]])
        ytr_n = np.hstack([yaug_n[: i], yaug_n[i + 1 :]])
        model.fit(Xtr_nxp, ytr_n)

        # -----compute w(x_i; z_{-i})-----
        # compute normalizing constant
        predall_n = model.predict(Xuniv_nxp)
        Z = np.sum(np.exp(lmbda * predall_n))

        testpred = model.predict(Xaug_nxp[i][None, :])
        ptest = np.exp(lmbda * testpred) / Z
        w_n1[i] = ptest / ptrain_fn(Xaug_nxp[i])

    return w_n1


def weighted_quantile(vals_n, weights_n, quantile, tol: float = 1e-12):
    if np.abs(np.sum(weights_n) - 1) > tol:
        raise ValueError("weights don't sum to one.")

    # sort values
    sorter = np.argsort(vals_n)
    vals_n = vals_n[sorter]
    weights_n = weights_n[sorter]

    cumweight_n = np.cumsum(weights_n)
    idx = np.searchsorted(cumweight_n, quantile, side='left')
    # idx = np.min(np.where(cumweight_n >= quantile)[0])
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
    return np.sum(np.abs(y - cs) < tol) > 0

def get_scores(model, Xaug_nxp, yaug_n):
    model.fit(Xaug_nxp, yaug_n)
    pred_n = model.predict(Xaug_nxp)
    scores_n1 = np.abs(yaug_n - pred_n)
    return scores_n1

def construct_covint_confidence_set_ridge(ys, Xtrain_nxp, ytrain_n, Xtest_1xp, ptrain_fn, Xuniv_uxp, gamma, lmbda,
                                           alpha: float = 0.1, print_every: int = 10, verbose: bool = True):
    """
    Special case of ridge regression model
    :param ys:
    :param Xtrain_nxp:
    :param ytrain_n:
    :param Xtest_1xp:
    :param ptrain_fn:
    :param Xuniv_nxp:
    :param gamma: ridge regularization parameter
    :param lmbda:
    :param alpha:
    :param print_every:
    :param verbose:
    :return:
    """
    np.set_printoptions(precision=3)
    n, p = Xtrain_nxp.shape
    Xaug_nxp = np.vstack([Xtrain_nxp, Xtest_1xp])

    # train n + 1 LOO models, store linear parameterizations of \mu_{-i, y}(X_i) as function of y
    ab_nx2 = np.zeros([n, 2])
    C_nxp = np.zeros([n, p])
    An_nxp = np.zeros([n, p])
    # t0 = time.time()
    for i in range(n):
        # construct A_{-i}
        Xi_nxp = np.vstack([Xaug_nxp[: i], Xaug_nxp[i + 1 :]]) # n rows
        invcov_pxp = sc.linalg.pinvh(Xi_nxp.T.dot(Xi_nxp) + gamma * np.eye(p))
        Ai = invcov_pxp.dot(Xi_nxp.T)  # p x n
        # cov_pxp = Xi_nxp.T.dot(Xi_nxp) + gamma * np.eye(p)  # not as fast as sc.linalg.pinvh
        # Ai = np.linalg.solve(cov_pxp, Xi_nxp.T)

        # compute linear parameterizations of \mu_{-i, y}(X_i)
        yi_ = np.hstack([ytrain_n[: i], ytrain_n[i + 1 :]])  # n - 1 elements
        Ci = Ai[:, : -1].dot(yi_) # p elements
        ai = Ci.dot(Xtrain_nxp[i])
        bi = Ai[:, -1].dot(Xtrain_nxp[i])

        # store
        ab_nx2[i] = ai, bi
        C_nxp[i] = Ci
        An_nxp[i] = Ai[:, -1]

    # i = n + 1 case
    model = Ridge(alpha=gamma)
    model.fit(Xtrain_nxp, ytrain_n)
    alast = model.predict(Xtest_1xp)  # a_{n + 1}

    # process scores for each candidate value y
    scores_n1xy = np.zeros([n + 1, ys.size])
    by_nxy = np.outer(ab_nx2[:, 1], ys)
    muhatiy_nxy = ab_nx2[:, 0][:, None] + by_nxy
    scores_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_nxy)
    scores_n1xy[-1] = np.abs(ys - alast)

    # process likelihood ratios for each candidate value y
    wi_num_nxy = np.exp(lmbda * muhatiy_nxy)
    betaiy_nxpxy = C_nxp[:, :, None] + ys * An_nxp[:, :, None]
    pred_nxyxu = np.tensordot(betaiy_nxpxy, Xuniv_uxp, axes=(1, 1))  # TODO: double-check
    normconst_nxy = np.sum(np.exp(lmbda * pred_nxyxu), axis=2)
    ptrain_n = ptrain_fn(Xtrain_nxp)

    wi_n1xy = np.zeros([n + 1, ys.size])
    wi_n1xy[: -1] = wi_num_nxy / (ptrain_n[:, None] * normconst_nxy)

    pred_u = model.predict(Xuniv_uxp)
    normconst = np.sum(np.exp(lmbda * pred_u))
    wi_n1xy[-1] = np.exp(lmbda * alast) / (ptrain_fn(Xtest_1xp) * normconst)

    q_y = get_quantile(alpha, wi_n1xy, scores_n1xy)
    cs = ys[scores_n1xy[-1] <= q_y]

    # process each candidate value y
    # scores_n1 = np.zeros([n + 1])
    # w_n1 = np.zeros([n + 1])
    # t0 = time.time()
    # for y_idx, y in enumerate(ys):
    #
    #     for i in range(n):
    #
    #         # compute score
    #         muhatiy = ab_nx2[i, 0] + ab_nx2[i, 1] * y
    #         Vi = np.abs(ytrain_n[i] - muhatiy)
    #         scores_n1[i] = Vi
    #
    #         # compute likelihood ratio
    #         wi_numerator = np.exp(lmbda * muhatiy)
    #         betaiy = C_nxp[i] + y * An_nxp[i]
    #         normconst = np.sum([np.exp(lmbda * betaiy.dot(X_p)) for X_p in Xuniv_nxp])
    #         wi = wi_numerator / (ptrain_fn(Xtrain_nxp[i]) * normconst)
    #         w_n1[i] = wi
    #
    #     # i = n + 1
    #     # score
    #     Vlast = np.abs(y - alast)
    #     scores_n1[-1] = Vlast
    #
    #     # likelihood ratio
    #     normconst = np.sum([np.exp(lmbda * model.predict(X_p[None, :])) for X_p in Xuniv_nxp])
    #     wlast = np.exp(lmbda * alast) / (ptrain_fn(Xtest_1xp) * normconst)
    #     w_n1[-1] = wlast
    #
    #     # check if add y to the confidence set
    #     q = get_quantile(alpha, w_n1, scores_n1)
    #     if scores_n1[-1] <= q:
    #         cs.append(y)
    #
    #     # print progress
    #     if verbose and (y_idx + 1) % print_every == 0:
    #         print("Done with {} / {} y values ({:.1f} s): {}".format(y_idx + 1, ys.size, time.time() - t0, np.array(cs)))

    return cs


def construct_covint_confidence_set(ys, Xtrain_nxp, ytrain_n, Xtest_p, ptrain_fn, Xuniv_nxp, model, lmbda,
                                    alpha: float = 0.1, print_every: int = 10, verbose: bool = True):
    # model needs fit() and predict() methods
    np.set_printoptions(precision=3)

    cs = []
    t0 = time.time()
    Xaug_nxp = np.vstack([Xtrain_nxp, Xtest_p])

    for y_idx, y in enumerate(ys):

        # get scores V_i^(x, y) for i = 1, ldots, n + 1
        yaug_n = np.hstack([ytrain_n, y])
        scores_n1 = get_scores(model, Xaug_nxp, yaug_n)

        # get w(x_i; z_{-i}) for i = 1, ..., n + 1
        w_n1 = get_lrs(Xaug_nxp, yaug_n, ptrain_fn, Xuniv_nxp, lmbda, model)

        # for each value of inverse temperature lambda, compute quantile of weighted scores
        q = get_quantile(alpha, w_n1, scores_n1)

        # if y <= quantile, include in confidence set
        if scores_n1[-1] <= q:
            cs.append(y)

        # print progress
        if verbose and (y_idx + 1) % print_every == 0:
            print("Done with {} / {} y values ({:.1f} s): {}".format(
                y_idx + 1, ys.size, time.time() - t0, np.array(cs)))
    return np.array(cs)



def construct_covshift_confidence_set(ys, Xtrain_nxp, ytrain_n, Xtest_p, ptrain_fn, model, Xuniv_nxp, lmbda,
                                    alpha: float = 0.1, print_every: int = 10):
    """
    :param ys: candidate y values for Xtest_p
    :param Xtrain_nxp:
    :param ytrain_n:
    :param Xtest_p:
    :param ptrain_fn: function that returns array of likelihood(s) for input n x p array of covariates
    :param model: already fitted! fit() and predict() methods
    :param lmbda: float, inverse temperature
    :param alpha: float, miscoverage
    :param print_every:
    :return: numpy array of y values
    """
    np.set_printoptions(precision=3)

    cs = []
    t0 = time.time()
    Xaug_nxp = np.vstack([Xtrain_nxp, Xtest_p])

    # get normalization constant for test covariate distribution
    predall_n = model.predict(Xuniv_nxp)
    punnorm_n = np.exp(lmbda * predall_n)
    Z = np.sum(punnorm_n)

    # get weights (likelihood ratios) for n + 1 covariates
    pred_n1 = model.predict(Xaug_nxp)  # need to call this first before refitting below for candidate y values!
    punnorm_n1 = np.exp(lmbda * pred_n1)
    ptest_n1 = punnorm_n1 / Z
    w_n1 = ptest_n1 / ptrain_fn(Xaug_nxp)

    for y_idx, y in enumerate(ys):

        # get scores
        yaug_n = np.hstack([ytrain_n, y])
        scores_n1 = get_scores(model, Xaug_nxp, yaug_n)  # TODO: can rewrite as linear function of y

        # compute quantile of weighted scores
        q = get_quantile(alpha, w_n1, scores_n1)

        # if y <= quantile, include in confidence set
        if scores_n1[-1] <= q:
            cs.append(y)

        # print progress
        if (y_idx + 1) % print_every == 0:
            print("Done with {} / {} y values ({:.1f} s): {}".format(
                y_idx + 1, ys.size, time.time() - t0, np.array(cs)))
    return np.array(cs)

