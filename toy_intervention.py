import numpy as np
import time
import scipy as sc

from sklearn.linear_model import Ridge

def get_wt_centered_train_data(wt_is_1, n_sample, p_mutate, seed: int = None):
    X_nxp = sc.stats.bernoulli.rvs(1 - p_mutate if wt_is_1 else p_mutate, size=(n_sample, 13))
    X_nxp[X_nxp == 0] = -1
    def ptrain_fn(Xtest_nxp):
        n_zero = np.sum(Xtest_nxp == -1, axis=1)
        n_one = Xtest_nxp.shape[1] - n_zero
        p_mutations = np.power(p_mutate, n_zero if wt_is_1 else n_one)
        p_wt = np.power(1 - p_mutate, n_one if wt_is_1 else n_zero)
        return p_mutations * p_wt
    return X_nxp, ptrain_fn

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
        w_n1[i] = ptest / ptrain_fn(Xaug_nxp[i][None, :])

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
    return np.any(np.abs(y - cs) < tol)

def get_scores(model, Xaug_nxp, yaug_n, use_loo_score: bool = False):
    if use_loo_score:
        scores_n1 = np.zeros([yaug_n.size])
        pred_n1 = np.zeros([yaug_n.size])
        for i in range(yaug_n.size):
            Xtr_nxp = np.vstack([Xaug_nxp[: i], Xaug_nxp[i + 1 :]])
            ytr_n = np.hstack([yaug_n[: i], yaug_n[i + 1 :]])
            model.fit(Xtr_nxp, ytr_n)
            pred_1 = model.predict(Xaug_nxp[i][None, :])
            scores_n1[i] = np.abs(yaug_n[i] - pred_1[0])
            pred_n1[i] = pred_1[0]
    else:
        model.fit(Xaug_nxp, yaug_n)
        pred_n1 = model.predict(Xaug_nxp)
        scores_n1 = np.abs(yaug_n - pred_n1)
    return scores_n1, pred_n1

def construct_covint_confidence_set_ridge(ys, Xtrain_nxp, ytrain_n, Xtest_1xp, ptrain_fn, Xuniv_uxp, gamma, lmbda,
                                           alpha: float = 0.1, use_lapack_inversion: bool = True):
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
    n, p = Xtrain_nxp.shape
    Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

    # for non-LOO score, only need to train one model for scores
    cov_pxp = Xaug_n1xp.T.dot(Xaug_n1xp) + gamma * np.eye(p)
    if use_lapack_inversion:
        # fastest way to invert PD matrices, but no robust error-checking
        # https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
        zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
        invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
        assert(info == 0)
        invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
    else:
        invcov_pxp = sc.linalg.pinvh(cov_pxp)
    A = invcov_pxp.dot(Xaug_n1xp.T)  # p x (n + 1)
    C = A[:, : -1].dot(ytrain_n)  # p elements

    a_n1 = C.dot(Xaug_n1xp.T)
    b_n1 = A[:, -1].dot(Xaug_n1xp.T)

    # train n + 1 LOO models, store linear parameterizations of \mu_{-i, y}(X_i) as function of y
    ab_nx2 = np.zeros([n, 2])
    C_nxp = np.zeros([n, p])
    An_nxp = np.zeros([n, p])
    for i in range(n):
        # construct A_{-i}
        Xi_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]]) # n rows
        cov_pxp = Xi_nxp.T.dot(Xi_nxp) + gamma * np.eye(p)

        # invert covariance matrix
        if use_lapack_inversion:
            zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
            invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
            assert(info == 0)
            invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
        else:
            invcov_pxp = sc.linalg.pinvh(cov_pxp)
        Ai = invcov_pxp.dot(Xi_nxp.T)  # p x n

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
    model = Ridge(alpha=gamma, fit_intercept=False)
    model.fit(Xtrain_nxp, ytrain_n)
    alast = model.predict(Xtest_1xp)  # a_{n + 1}

    # process scores for each candidate value y
    scoresloo_n1xy = np.zeros([n + 1, ys.size])
    by_nxy = np.outer(ab_nx2[:, 1], ys)
    muhatiy_nxy = ab_nx2[:, 0][:, None] + by_nxy
    scoresloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_nxy)
    scoresloo_n1xy[-1] = np.abs(ys - alast)

    scoresnoloo_n1xy = np.zeros([n + 1, ys.size])
    by_n1xy = np.outer(b_n1, ys)
    muhatiy_n1xy = a_n1[:, None] + by_n1xy
    scoresnoloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_n1xy[: -1])
    scoresnoloo_n1xy[-1] = np.abs(ys - muhatiy_n1xy[-1])

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

    # construct confidence set
    q_y = get_quantile(alpha, wi_n1xy, scoresloo_n1xy)
    loocs = ys[scoresloo_n1xy[-1] <= q_y]
    q_y = get_quantile(alpha, wi_n1xy, scoresnoloo_n1xy)
    noloocs = ys[scoresnoloo_n1xy[-1] <= q_y]
    return loocs, noloocs, scoresloo_n1xy, scoresnoloo_n1xy, wi_n1xy


def construct_covint_confidence_set(ys, Xtrain_nxp, ytrain_n, Xtest_p, ptrain_fn, Xuniv_nxp, model, lmbda,
                                    alpha: float = 0.1, use_loo_score: bool = False,
                                    print_every: int = 10, verbose: bool = True):
    # model needs fit() and predict() methods
    np.set_printoptions(precision=3)

    cs = []
    t0 = time.time()
    n = ytrain_n.size
    Xaug_nxp = np.vstack([Xtrain_nxp, Xtest_p])
    scores_n1xy, wi_n1xy = np.zeros([n + 1, ys.size]), np.zeros([n + 1, ys.size])

    for y_idx, y in enumerate(ys):

        # get scores V_i^(x, y) for i = 1, ldots, n + 1
        yaug_n = np.hstack([ytrain_n, y])
        scores_n1 = get_scores(model, Xaug_nxp, yaug_n, use_loo_score=use_loo_score)
        scores_n1xy[:, y_idx] = scores_n1

        # get w(x_i; z_{-i}) for i = 1, ..., n + 1
        wi_n1 = get_lrs(Xaug_nxp, yaug_n, ptrain_fn, Xuniv_nxp, lmbda, model)
        wi_n1xy[:, y_idx] = wi_n1

        # for each value of inverse temperature lambda, compute quantile of weighted scores
        q = get_quantile(alpha, wi_n1, scores_n1)

        # if y <= quantile, include in confidence set
        if scores_n1[-1] <= q:
            cs.append(y)

        # print progress
        if verbose and (y_idx + 1) % print_every == 0:
            print("Done with {} / {} y values ({:.1f} s): {}".format(
                y_idx + 1, ys.size, time.time() - t0, np.array(cs)))
    return np.array(cs), scores_n1xy, wi_n1xy


def construct_covshift_confidence_set_ridge(ys, Xtrain_nxp, ytrain_n, Xtest_1xp, ptrain_fn, Xuniv_nxp, gamma, lmbda,
                                            alpha: float = 0.1, use_lapack_inversion: bool = True):

    n, p = Xtrain_nxp.shape
    Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
    model = Ridge(alpha=gamma, fit_intercept=False)
    model.fit(Xtrain_nxp, ytrain_n)

    # ==== compute likelihood ratios =====
    # get normalization constant for test covariate distribution
    predall_n = model.predict(Xuniv_nxp)
    Z = np.sum(np.exp(lmbda * predall_n))

    # get weights (likelihood ratios) for n + 1 covariates
    pred_n1 = model.predict(Xaug_n1xp)
    ptest_n1 = np.exp(lmbda * pred_n1) / Z
    wi_n1 = ptest_n1 / ptrain_fn(Xaug_n1xp)

    # ===== compute scores =====

    # for non-LOO score, only need to train one model for scores
    cov_pxp = Xaug_n1xp.T.dot(Xaug_n1xp) + gamma * np.eye(p)
    zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
    invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
    assert(info == 0)
    invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
    A = invcov_pxp.dot(Xaug_n1xp.T)  # p x (n + 1)
    C = A[:, : -1].dot(ytrain_n)  # p elements

    a_n1 = C.dot(Xaug_n1xp.T)
    b_n1 = A[:, -1].dot(Xaug_n1xp.T)

    # train n + 1 LOO models, store linear parameterizations of \mu_{-i, y}(X_i) as function of y
    ab_nx2 = np.zeros([n, 2])
    C_nxp = np.zeros([n, p])
    An_nxp = np.zeros([n, p])
    for i in range(n):
        # construct A_{-i}
        Xi_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]]) # n rows
        cov_pxp = Xi_nxp.T.dot(Xi_nxp) + gamma * np.eye(p)
        if use_lapack_inversion:
            zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
            invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
            assert(info == 0)
            invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
        else:
            invcov_pxp = sc.linalg.pinvh(cov_pxp)
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
    alast = model.predict(Xtest_1xp)  # a_{n + 1}

    # compute scores for each candidate value y
    scoresloo_n1xy = np.zeros([n + 1, ys.size])
    by_nxy = np.outer(ab_nx2[:, 1], ys)
    muhatiy_nxy = ab_nx2[:, 0][:, None] + by_nxy
    scoresloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_nxy)
    scoresloo_n1xy[-1] = np.abs(ys - alast)

    scoresnoloo_n1xy = np.zeros([n + 1, ys.size])
    by_n1xy = np.outer(b_n1, ys)
    muhatiy_n1xy = a_n1[:, None] + by_n1xy
    scoresnoloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_n1xy[: -1])
    scoresnoloo_n1xy[-1] = np.abs(ys - muhatiy_n1xy[-1])

    # keep y values that fall under weighted quantile
    wi_n1xy = wi_n1[:, None] * np.ones([n + 1, ys.size])
    q_y = get_quantile(alpha, wi_n1xy, scoresloo_n1xy)
    loocs = ys[scoresloo_n1xy[-1] <= q_y]
    q_y = get_quantile(alpha, wi_n1xy, scoresnoloo_n1xy)
    noloocs = ys[scoresnoloo_n1xy[-1] <= q_y]
    return loocs, noloocs



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
    scoresnoloo_n1xy = np.zeros([Xaug_nxp.shape[0], ys.size])

    for y_idx, y in enumerate(ys):

        # get scores
        yaug_n = np.hstack([ytrain_n, y])
        scores_n1 = get_scores(model, Xaug_nxp, yaug_n)
        scoresnoloo_n1xy[:, y_idx] = scores_n1

        # compute quantile of weighted scores
        q = get_quantile(alpha, w_n1, scores_n1)

        # if y <= quantile, include in confidence set
        if scores_n1[-1] <= q:
            cs.append(y)

        # print progress
        if (y_idx + 1) % print_every == 0:
            print("Done with {} / {} y values ({:.1f} s): {}".format(
                y_idx + 1, ys.size, time.time() - t0, np.array(cs)))
    return np.array(cs), scoresnoloo_n1xy

