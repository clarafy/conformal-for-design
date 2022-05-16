import numpy as np
import time
import scipy as sc

import calibrate as cal

from tensorflow import keras

from aav.entropy_opt import normalize_theta, featurize
from aav.opt_analysis import aa_probs_from_nuc_probs
from aav.data_prep import one_hot_encode
from aav.modeling import DataGenerator

# NNK distribution
tmp = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0, 0.5, 0, 0.5]])
pnnknuc_lxk = np.tile(tmp, [7, 1])
pnnkaa_lxk = np.array(aa_probs_from_nuc_probs(pnnknuc_lxk).T)

ddata = np.load('../aav/data.npz')
seqs_n = ddata['seqs_n']
fit_nx2 = ddata['fit_nx2']
didx = np.load('../aav/models/500000_idx.npz')

# ========== rejection sampling from test distribution ==========

def thetanuc2paa(thetanuc_lxk):
    pnuc_lxk = normalize_theta(thetanuc_lxk)
    paa_lxk = np.array(aa_probs_from_nuc_probs(pnuc_lxk).T)
    return paa_lxk

def get_loglikelihood(ohe_nxlxk, theta_lxk, normalized: bool = True):
    if normalized:  #  treat theta_lxk as p_lxk
        logp_1xlxk = np.log(theta_lxk)[None, :, :]
        logp_n = np.sum(ohe_nxlxk * logp_1xlxk, axis=(1, 2))
    else:
        exptheta_lxk = np.exp(theta_lxk)
        logp_n = theta_lxk - np.log(np.sum(exptheta_lxk, axis=1))
    return logp_n

def sample_from_test_distribution(thetatestnuc_lxk, test_idx, ptrainaa_lxk = None):

    if ptrainaa_lxk is None:
        ptrainaa_lxk = pnnkaa_lxk.copy()

    testseqs_n = seqs_n[test_idx]

    oheaa_lxk_list = [one_hot_encode(seq, flatten=False) for seq in testseqs_n]
    ohetrainaa_nxlxk = np.stack(oheaa_lxk_list)

    # compute training likelihood for each covariate
    logptrain_n = get_loglikelihood(ohetrainaa_nxlxk, ptrainaa_lxk, normalized=True)

    # compute test likelihood for each covariate
    ptestaa_lxk = thetanuc2paa(thetatestnuc_lxk)
    logptest_n = get_loglikelihood(ohetrainaa_nxlxk, ptestaa_lxk, normalized=True)

    # compute M as upper bound on max_x test(x) / train(x)
    ratio_lxk = ptestaa_lxk / ptrainaa_lxk
    maxp_l = np.max(ratio_lxk, axis=1)
    M = np.prod(maxp_l)

    # compute acceptance probability for each test data point = test(x) / (M * train(x)) = exp(log(test(x) - (logM + log(train(x))))
    # randomly accept or reject each test data point
    paccept_n = np.exp(logptest_n - (np.log(M) + logptrain_n))
    acc_n = sc.stats.bernoulli.rvs(paccept_n)
    samp_idx = np.where(acc_n > 0)[0]
    return testseqs_n[samp_idx], samp_idx, paccept_n

def get_coverage(lq_nx2, fit_n):
    cov = np.sum((fit_n >= lq_nx2[:, 0]) & (fit_n <= lq_nx2[:, 1])) / fit_n.size
    return cov

def evaluate_split_conformal_aav(thetatestnuc_lxlxk, n_trial: int = 10, n_cal: int = 500000, alpha: float = 0.1):
    n_lambda = len(thetatestnuc_lxlxk)

    # load all calibration and test data, merge
    idx = np.hstack([didx['cal_idx'], didx['test_idx']])
    seqs_m = seqs_n[idx]
    m = idx.size
    print('Loaded {} calibration data points'.format(m))

    # compute point predictions and scores on all data
    datagen = DataGenerator(seqs_m, np.zeros([m, 2]), range(m), one_hot_encode, batch_size=10000, shuffle=False)
    # cal_gen = featurize(seqs_m, one_hot_encode, aa=False)  # HERE: bug
    pred_m = np.zeros([m])
    for seed in range(5):
        model = keras.models.load_model('../aav/models/nn100_is_seed{}_011122.npy'.format(seed))
        pred_m += model.predict_generator(datagen).reshape(m)
    pred_m /= 5
    score_m = np.abs(pred_m - fit_nx2[idx, 0])

    # compute training likelihoods of all data
    tmp = [one_hot_encode(seq, flatten=False) for seq in seqs_m]
    oheaa_mxlxk = np.stack(tmp)
    logptrain_m = get_loglikelihood(oheaa_mxlxk, pnnkaa_lxk, normalized=True)

    cov_lxt = np.zeros([n_lambda, n_trial])
    avglen_lxt = np.zeros([n_lambda, n_trial])
    len_lxt = {(l_idx, t): None for l_idx, t in zip(range(n_lambda), range(n_trial))}
    fit_lxt = {(l_idx, t): None for l_idx, t in zip(range(n_lambda), range(n_trial))}
    for l_idx in range(n_lambda):
        t0 = time.time()

        ptestaa_lxk = thetanuc2paa(thetatestnuc_lxlxk[l_idx])
        ratio_lxk = ptestaa_lxk / pnnkaa_lxk  # do in theta space?
        maxp_l = np.max(ratio_lxk, axis=1)
        M = np.prod(maxp_l)

        # compute test likelihoods of all data
        logptest_m = get_loglikelihood(oheaa_mxlxk, ptestaa_lxk, normalized=True)
        paccept_m = np.exp(logptest_m - (np.log(M) + logptrain_m))

        # compute (unnormalized) weights for all data
        w_m = np.exp(logptest_m - logptrain_m)

        for t in range(n_trial):
            # partition off calibration data
            shuffle_idx = np.random.permutation(m)
            cal_idx, test_idx = shuffle_idx[: n_cal], shuffle_idx[n_cal :]

            # sample from test distribution
            acc_n = sc.stats.bernoulli.rvs(paccept_m[test_idx])
            testsamp_idx = np.where(acc_n > 0)[0]

            pred_n = pred_m[test_idx[testsamp_idx]]
            n_test = testsamp_idx.size
            print("{} samples from test distribution.".format(n_test))

            # construct confidence intervals
            lq_nx2 = np.zeros([n_test, 2])
            p_nxm1 = np.hstack([np.tile(w_m[cal_idx][None, :], [n_test, 1]), w_m[test_idx[testsamp_idx]][:, None]])
            p_nxm1 /= np.sum(p_nxm1, axis=1, keepdims=True)
            avglen = 0
            len_t = np.zeros([n_test])
            for i in range(n_test):
                # compute quantile of discrete distribution of weighted calibration scores
                augscore_m1 = np.hstack([score_m[cal_idx], [np.infty]])
                q = cal.weighted_quantile(augscore_m1, p_nxm1[i], 1 - alpha)
                lq_nx2[i] = pred_n[i] - q, pred_n[i] + q
                avglen += 2 * q
                len_t[i] = 2 * q

            fit = fit_nx2[idx[test_idx[testsamp_idx]], 0]
            cov = get_coverage(lq_nx2, fit)
            cov_lxt[l_idx, t] = cov
            avglen /= n_test
            avglen_lxt[l_idx, t] = avglen
            len_lxt[(l_idx, t)] = len_t
            fit_lxt[(l_idx, t)] = fit
        print("  {:.4f}, {:.2f}, {:.1f} s".format(np.mean(cov_lxt[l_idx]), np.mean(avglen_lxt[l_idx]), time.time() - t0))
    return cov_lxt, len_lxt, avglen_lxt, fit_lxt

class SplitConformalCovariateShift(object):
    def __init__(self, pcalaa_lxk: np.array = None):

        cal_idx = didx['cal_idx']
        m = cal_idx.size
        self.calseq_m = seqs_n[cal_idx]
        print('Loaded {} calibration data points'.format(m))

        cal_gen = DataGenerator(self.calseq_m, fit_nx2, range(m), one_hot_encode, batch_size=10000, shuffle=False)
        # cal_gen = featurize(self.calseq_m, one_hot_encode, aa=False)
        pred_m = np.zeros([m])
        self.models = []
        # load ensemble of trained models, compute predictions on calibration data
        for seed in range(5):
            model = keras.models.load_model('../aav/models/nn100_is_seed{}_011122.npy'.format(seed))
            pred_m += model.predict_generator(cal_gen).reshape(m)
            self.models.append(model)
        pred_m /= 5

        # compute scores on calibration data
        self.calscores_m = np.abs(pred_m - fit_nx2[cal_idx, 0])

        # compute training likelihoods of calibration data
        tmp = [one_hot_encode(seq, flatten=False) for seq in self.calseq_m]
        self.ohecalaa_mxlxk = np.stack(tmp)
        self.pcalaa_lxk = pnnkaa_lxk.copy() if pcalaa_lxk is None else pcalaa_lxk
        self.logptrain_m = get_loglikelihood(self.ohecalaa_mxlxk, self.pcalaa_lxk, normalized=True)

    def fit(self, alpha: float, testsamp_idx, thetatestnuc_lxk):

        n = testsamp_idx.size
        testseqs_n = seqs_n[testsamp_idx]
        testfit_nx2 = fit_nx2[testsamp_idx]  # not strictly used
        test_gen = DataGenerator(testseqs_n, testfit_nx2, range(n), one_hot_encode, batch_size=testseqs_n.size, shuffle=False)
        # test_gen = featurize(testseqs_n, one_hot_encode, AA=False)

        # assuming nucleotide-based theta_test
        ptestaa_lxk = thetanuc2paa(thetatestnuc_lxk)

        # compute test likelihoods and weights of calibration covariates
        logptest_m = get_loglikelihood(self.ohecalaa_mxlxk, ptestaa_lxk, normalized=True)
        wcal_m = np.exp(logptest_m - self.logptrain_m)

        # compute weights for test covariates
        oheaa_lxk_list = [one_hot_encode(seq, flatten=False) for seq in testseqs_n]
        testoheaa_nxlxk = np.stack(oheaa_lxk_list)
        testlogpcal_n = get_loglikelihood(testoheaa_nxlxk, self.pcalaa_lxk)
        testlogptest_n = get_loglikelihood(testoheaa_nxlxk, ptestaa_lxk)
        wtest_n = np.exp(testlogptest_n - testlogpcal_n)

        # for each test covariate, point prediction
        pred_n = np.zeros([n])
        # load ensemble of trained models, compute predictions on calibration data
        for seed in range(5):
            pred_n += self.models[seed].predict_generator(test_gen).reshape(n)
        pred_n /= 5

        # for each test covariate, produce confidence set
        lq_nx2 = np.zeros([n, 2])
        p_nxm1 = np.hstack([np.tile(wcal_m[None, :], [n, 1]), wtest_n[:, None]])
        p_nxm1 /= np.sum(p_nxm1, axis=1, keepdims=True)
        for i in range(n):
            # compute quantile of discrete distribution of weighted calibration scores
            q = cal.weighted_quantile(self.calscores_m, p_nxm1[i], 1 - alpha)
            lq_nx2[i] = pred_n[i] - q, pred_n[i] + q

        cov = get_coverage(lq_nx2, testfit_nx2[:, 0])
        return lq_nx2, pred_n, cov
