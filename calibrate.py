from abc import ABC, abstractmethod
import time

import numpy as np
import scipy as sc

import util
from assay import Assay

from sklearn.linear_model import Ridge, RidgeCV


class FitnessModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X_nxp, y_n, **kwargs):
        pass

    @abstractmethod
    def get_next_sequences(self, X_nxp, k: int):
        pass

    @abstractmethod
    def get_uncertainty(self, X_nxp, idx):
        pass

class TopKRidgeRegression(FitnessModel):
    def __init__(self, fit_intercept: bool = True, reg = 1e-6, regs = None):
        if regs is not None:
            self.model = RidgeCV(fit_intercept=fit_intercept, alphas=regs)
        else:
            self.model = Ridge(fit_intercept=fit_intercept, alpha=reg)
        self.fit_intercept = fit_intercept
        # self.reg = reg

    def fit(self, X_nxp, y_n, **kwargs):
        self.model.fit(X_nxp, y_n, **kwargs)
        self.reg = self.model.alpha_

    def predict(self, X_nxp):
        return self.model.predict(X_nxp)

    def get_next_sequences(self, X_nxp, k: int, exclude_idx = None):
        if exclude_idx is None:
            exclude_idx = []
        pred_n = self.model.predict(X_nxp)
        best_idx = np.argsort(pred_n)[::-1]
        # ignore samples already observed before
        delete_idx = [np.where(best_idx == i)[0] for i in exclude_idx]
        idx = np.delete(best_idx, delete_idx)
        return idx[: k]

    def get_uncertainty(self, X_nxp, newidx_k, oldidx, uncertainty_type: str = 'inverse_prediction'):
        if uncertainty_type == "confidence_ellipsoid":
            if self.fit_intercept:
                X_nxp = np.hstack([np.ones((X_nxp.shape[0], 1)), X_nxp])
            xa_kxp = X_nxp[newidx_k]
            V_pxp = X_nxp[oldidx].T.dot(X_nxp[oldidx]) + self.reg * np.eye(X_nxp.shape[1])
            Vinv_pxp = np.linalg.inv(V_pxp)
            return np.array([np.sqrt(xa_p.dot(Vinv_pxp.dot(xa_p))) for xa_p in xa_kxp])
        elif uncertainty_type == 'inverse_prediction':
            return 1. / np.exp(self.model.predict(X_nxp[newidx_k]))
        return np.ones([newidx_k.size])

def qhatplus(vals_kxn, alpha):
    k, n = vals_kxn.shape
    if alpha < 1 / (n + 1):
        return np.inf * np.ones([k])
    idx = int(np.ceil((1 - alpha) * (n + 1)))
    return np.sort(vals_kxn, axis=1)[:, idx - 1]

def qhatminus(vals_kxn, alpha):
    idx = int(np.floor(alpha * (vals_kxn.shape[1] + 1)))
    return np.sort(vals_kxn, axis=1)[:, idx - 1]

def jackknife(model, alpha: float, X_nxp, arms_sxtxk, reward_sxtxk, pred_sxtxk, u_sxtxk,
              uncertainty_type: str = "inverse_prediction", print_every: int = 10):
    n_seed, tmp, k = reward_sxtxk.shape
    n_round = tmp - 1
    score_sxt = np.zeros([n_seed, n_round + 1])
    jkpluscov_sxtxk = np.zeros([n_seed, n_round + 1, k])
    jkcov_sxtxk = np.zeros([n_seed, n_round + 1, k])
    jklen_sxtxk = np.zeros([n_seed, n_round +1, k])
    jkplen_sxtxk = np.zeros([n_seed, n_round +1, k])
    score_sxt[:, 0] = np.nan
    jkpluscov_sxtxk[:, 0] = np.nan
    jkcov_sxtxk[:, 0] = np.nan
    jklen_sxtxk[:, 0] = np.nan
    jkplen_sxtxk[:, 0] = np.nan
    t0 = time.time()

    for seed in range(n_seed):
        for t in range(1, n_round + 1):
            # gather rewards from all previous rounds as training data
            Xtrain_nxp = X_nxp[arms_sxtxk[seed, : t].flatten()]
            ytrain_n = reward_sxtxk[seed, : t].flatten()
            n_train = ytrain_n.size
            score_ = np.zeros([n_train])
            pluslo_kx, plushi_kx = np.zeros([k, n_train]), np.zeros([k, n_train])

            # LOO scores
            for i in range(n_train):
                tr_idx = np.hstack([np.arange(i), np.arange(i + 1, n_train)])
                Xtr_nxp, ytr_n = Xtrain_nxp[tr_idx], ytrain_n[tr_idx]
                Xte_nxp = Xtrain_nxp[i][None, :]
                ute_n = model.get_uncertainty(X_nxp, np.array([i]), None, uncertainty_type=uncertainty_type)
                model.fit(Xtr_nxp, ytr_n)  # LOO model mu_{-i}
                score_[i] = np.abs(ytrain_n[i] - model.predict(Xte_nxp)) / ute_n # LOO score
                loopred_k = model.predict(X_nxp[arms_sxtxk[seed, t].flatten()])  # \hat{mu}_{-i}(X_{n + 1})
                pluslo_kx[:, i] = loopred_k - score_[i] * u_sxtxk[seed, t]
                plushi_kx[:, i] = loopred_k + score_[i] * u_sxtxk[seed, t]

            # take quantile of scores
            score_sxt[seed, t] = np.quantile(score_, 1 - alpha)

            # coverage for jackknife interval
            lbcov_k = reward_sxtxk[seed, t] >= pred_sxtxk[seed, t] - score_sxt[seed, t] * u_sxtxk[seed, t]
            ubcov_k = reward_sxtxk[seed, t] <= pred_sxtxk[seed, t] + score_sxt[seed, t] * u_sxtxk[seed, t]
            jkcov_sxtxk[seed, t] = lbcov_k * ubcov_k
            jklen_sxtxk[seed, t] = 2 * score_sxt[seed, t] * u_sxtxk[seed, t]

            # coverage for jackknife+ interval
            lb_k = qhatminus(pluslo_kx, alpha)
            ub_k = qhatplus(plushi_kx, alpha)
            lbcov_k = reward_sxtxk[seed, t] >= lb_k
            ubcov_k = reward_sxtxk[seed, t] <= ub_k
            jkpluscov_sxtxk[seed, t] = lbcov_k * ubcov_k
            jkplen_sxtxk[seed, t, :] = ub_k - lb_k

        if (seed + 1) % print_every == 0:
            jkpluscov = np.mean(jkpluscov_sxtxk[: seed + 1, -1])
            jkcov = np.mean(jkcov_sxtxk[: seed + 1, -1])
            print("Done with {} / {} seeds. Final JK, JK+ coverage: {:.4f}, {:.4f}. {:.1f} s".format(
                seed + 1, n_seed, jkcov, jkpluscov, time.time() - t0))
    return score_sxt, jkcov_sxtxk, jkpluscov_sxtxk, jklen_sxtxk, jkplen_sxtxk


def calibrate(alpha: float, reward_sxtxk, pred_sxtxk, u_sxtxk, signed: bool = True, conformal_window: int = 50):
    n_seed, tmp, k = reward_sxtxk.shape
    n_round = tmp - 1

    # compute calibrated intervals
    signedres_sxtxk = reward_sxtxk - pred_sxtxk # [:, 0, :] will be np.nan because no predictions
    interval_sxtxkx2 = np.zeros([n_seed, n_round + 1, k, 2])
    interval_sxtxkx2[:, : 2] = np.nan  # only have data to make intervals for t = 2 onward
    miscov_sxtxkx2 = np.zeros([n_seed, n_round + 1, k, 2])
    miscov_sxtxkx2[:, 0] = np.nan
    score_sxtxk = signedres_sxtxk / u_sxtxk

    for seed in range(n_seed):
        for t in range(2, n_round + 1):

            # compute intervals
            # if t == 2:
            #     t0 = 1
            # else:
            t0 = np.max([1, t - conformal_window])
            if signed:
                s_txk = signedres_sxtxk[seed, t0 : t] / u_sxtxk[seed, t0 : t]
                qlow = np.quantile(s_txk, alpha / 2)  # does not treat the k guys differently
                qhigh = np.quantile(s_txk, 1 - (alpha / 2))
                interval_sxtxkx2[seed, t, :, 0] = pred_sxtxk[seed, t] + qlow * u_sxtxk[seed, t]
                interval_sxtxkx2[seed, t, :, 1] = pred_sxtxk[seed, t] + qhigh * u_sxtxk[seed, t]
            else:
                s_txk = np.abs(signedres_sxtxk[seed, t0 : t]) / u_sxtxk[seed, t0 : t]
                q = np.quantile(s_txk, 1 - alpha)  # does not treat the k guys differently
                interval_sxtxkx2[seed, t, :, 0] = pred_sxtxk[seed, t] - q * u_sxtxk[seed, t]
                interval_sxtxkx2[seed, t, :, 1] = pred_sxtxk[seed, t] + q * u_sxtxk[seed, t]

            # check miscoverage
            miscov_sxtxkx2[seed, t, :, 0] = reward_sxtxk[seed, t] < interval_sxtxkx2[seed, t, :, 0]
            miscov_sxtxkx2[seed, t, :, 1] = reward_sxtxk[seed, t] > interval_sxtxkx2[seed, t, :, 1]
    return interval_sxtxkx2, miscov_sxtxkx2, score_sxtxk



def design_and_calibrate(model: FitnessModel, data: Assay, alpha: float = 0.1, k: int = 1, n_round: int = 200,
                         initial_idx: np.array = None, exclude_observed: bool = True, n_seed: int = 1000,
                         conformal_window: int = 50, print_every_seed: int = 10, print_every_it: int = 100,
                         uncertainty_type: str = "inverse_prediction", signed_residuals: bool = False):
    arms_sxtxk = np.zeros([n_seed, n_round + 1, k])
    reward_sxtxk = np.zeros([n_seed, n_round + 1, k])
    pred_sxtxk = np.zeros([n_seed, n_round + 1, k])
    pred_sxtxk[:, 0, :] = np.nan  # no predictions for initial training data at t = 0
    u_sxtxk = np.zeros([n_seed, n_round + 1, k])
    u_sxtxk[:, 0, :] = np.nan  # no notion of uncertainty for initial training data
    intlb_sxtxk = np.zeros([n_seed, n_round + 1, k])
    rmse_sxt = np.zeros([n_seed, n_round + 1])  #  RMSE on entire space
    rmse_sxt[:, 0] = np.nan

    if initial_idx is None:
        initial_idx = np.arange(data.n)

    time0 = time.time()
    # run design
    for seed in range(n_seed):

        # select initial training sequence(s) and get measurements
        arms_k = np.random.choice(initial_idx, size=k, replace=False)
        arms_sxtxk[seed, 0] = arms_k
        y_n = data.get_measurements(arms_k)
        reward_sxtxk[seed, 0] = y_n
        intlb_2xk = np.array(sc.stats.norm.interval(1 - alpha, scale=data.se_n[arms_k]))
        intlb_sxtxk[seed, 0] = intlb_2xk[1] - intlb_2xk[0]

        for t in range(1, n_round + 1):

            # (re)fit model, design new sequences, get model uncertainty on them
            model.fit(data.X_nxp[arms_k], y_n)

            # RMSE on all unseen proteins
            test_idx = np.delete(np.arange(data.y_n.size), arms_k)
            pred_n = model.predict(data.X_nxp[test_idx])
            rmse = np.sqrt(np.mean(np.square(pred_n - data.y_n[test_idx])))
            rmse_sxt[seed, t] = rmse

            newarms_k = model.get_next_sequences(data.X_nxp, k, exclude_idx=arms_k if exclude_observed else None)
            arms_sxtxk[seed, t] = newarms_k
            u_sxtxk[seed, t] = model.get_uncertainty(data.X_nxp, newarms_k, arms_k, uncertainty_type=uncertainty_type)
            intlb_2xk = np.array(sc.stats.norm.interval(1 - alpha, scale=data.se_n[newarms_k]))
            intlb_sxtxk[seed, t] = intlb_2xk[1] - intlb_2xk[0]

            # make predictions on new sequences
            pred_sxtxk[seed, t] = model.predict(data.X_nxp[newarms_k])

            # get measurements on new sequences
            ynew_n = data.get_measurements(newarms_k)
            reward_sxtxk[seed, t] = ynew_n

            # concatenate arms and rewards
            arms_k = np.hstack([arms_k, newarms_k])
            y_n = np.hstack([y_n, ynew_n])

            if ((seed + 1) % print_every_seed == 0) and (t % print_every_it == 0):
                print("Iter {}. Mean reward: {:.3f}".format(t, np.mean(ynew_n)))

        if (seed + 1) % print_every_seed == 0:
            print("Done with seed {} /  {} ({:.1f} s). Max reward: {:.3f}".format(
                seed + 1, n_seed, time.time() - time0, np.max(reward_sxtxk[seed])))

    # compute calibrated intervals
    print("Computing intervals...")
    interval_sxtxkx2, coverage_sxtxkx2, score_sxtxk = calibrate(
        alpha, reward_sxtxk, pred_sxtxk, u_sxtxk, signed=signed_residuals, conformal_window=conformal_window)
    print("Done ({:.1f} s)".format(time.time() - time0))
    return pred_sxtxk, reward_sxtxk, u_sxtxk, interval_sxtxkx2, coverage_sxtxkx2, intlb_sxtxk, arms_sxtxk.astype(int), score_sxtxk, rmse_sxt

