import numpy as np
import time
import scipy as sc



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


def get_quantile(alpha, w_n1, scores_n1):
    """
    Returns weighted 1 - alpha quantile of n + 1 scores, given likelihood ratios w(x_i; z_{-i}) and scores
    :param alpha: float, miscoverage level
    :param w_n1: numpy array, n + 1 likelihood ratios
    :param scores_n1: numpy array, n + 1 scores
    :return: float
    """
    p_n1 = w_n1 / np.sum(w_n1)
    augscores_n1 = np.hstack([scores_n1[: -1], np.inf])
    q = weighted_quantile(augscores_n1, p_n1, 1 - alpha)
    return q

def is_covered(y, cs, tol):
    return np.sum(np.abs(y - cs) < tol) > 0

def get_scores(model, Xaug_nxp, yaug_n):
    model.fit(Xaug_nxp, yaug_n)
    pred_n = model.predict(Xaug_nxp)
    scores_n1 = np.abs(yaug_n - pred_n)
    return scores_n1




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
        scores_n1 = get_scores(model, Xaug_nxp, yaug_n)

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

