from abc import ABC, abstractmethod
import time
import parse

import numpy as np
import scipy as sc
import pandas as pd

from sklearn.linear_model import LinearRegression
import util

HIS3P_ALPHASZS = [2, 2, 3, 2, 2, 3, 3, 4, 2, 4, 4]

class Assay(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_measurements(self, x_idx: np.array, seed: int = None):
        raise NotImplementedError

class PoelwijkData(Assay):
    fitness_names = ['red', 'blue']

    def __init__(self, fitness: str, order: int = 1, noise_scale: float = 1., append_distance: bool = False,
                 noise_estimate_order: int = 7, sig_level: float = 0.01,
                 load_precomputed_noise: bool = True):
        if fitness not in self.fitness_names:
            raise ValueError('Unrecognized fitness name: {}'.format(fitness))

        # ===== featurize sequences as higher-order interactions =====
        df = self.read_poelwijk_supp3()
        self.Xsigned_nxp = self.strarr2signedarr(df.binary_genotype)  # 1/-1 encoding of sequences
        self.X_nxp = util.walsh_hadamard_from_seqs(self.Xsigned_nxp, order=order) # featurize including intercept

        # normalizing all non-intercept features: WHT already orthonormal basis!
        # Xstd_1xp = np.std(self.X_nxp, axis=0, keepdims=True)
        # self.X_nxp[:, 1 :] = self.X_nxp[:, 1 :] / Xstd_1xp[:, 1 :]

        # ESM-1v masked marginal as feature
        # d = np.load('../poelwijk/esm_mm.npz')
        # feat_n = np.mean(d['score_nx5'], axis=1)
        # self.X_nxp = np.hstack([self.X_nxp, feat_n[:, None]])

        self.n, self.p = self.X_nxp.shape
        self.order = order
        print('{} features'.format(self.p))

        # ===== compute fitness and estimate measurement noise =====
        if fitness == 'blue':
            self.y_n = np.array(df.brightness_blue)
        elif fitness == 'red':
            self.y_n = np.array(df.brightness_red)

        # ===== estimate unexplainable noise via WHT =====
        if load_precomputed_noise:
            d = np.load('../poelwijk/{}_noise.npz'.format(fitness))
            self.se_n = d['se_n']
            self.noise_scale = noise_scale
            print("Loading estimated SE precomputed with order {} and significance level {}".format(
                d['order_est_noise'], d['sig_level']))
        else:
            t0 = time.time()
            X_nxp = util.walsh_hadamard_from_seqs(self.Xsigned_nxp, order=noise_estimate_order)
            n_term = X_nxp.shape[1]
            print('Estimating noise using {} interaction terms up to order {}'.format(n_term, noise_estimate_order))
            ols = LinearRegression(fit_intercept=False)
            ols.fit(X_nxp, self.y_n)
            pred_n = ols.predict(X_nxp)
            mse = np.sum(np.square(self.y_n - pred_n)) / (self.n - n_term)

            var_b = mse * (np.linalg.inv(np.dot(X_nxp.T, X_nxp)).diagonal())
            sd_b = np.sqrt(var_b)
            ts_b = ols.coef_ / sd_b

            pvals = np.array([2 * (1 - sc.stats.t.cdf(np.abs(i), (self.n - n_term))) for i in ts_b])
            threshold = 1 - np.power(1 - sig_level, 1 / n_term)
            sigterm_idx = np.where(pvals < threshold)[0]

            print("{} significant terms below {} for level {}. {:.1f} s".format(
                sigterm_idx.size, threshold, sig_level, time.time() - t0))
            pred_n = X_nxp[:, sigterm_idx].dot(ols.coef_[sigterm_idx])
            self.coef_pvals = pvals

            self.se_n = np.abs(pred_n - self.y_n)
            self.noise_scale = noise_scale
            np.savez('../poelwijk/{}_noise.npz'.format(fitness),
                     se_n=self.se_n, noise_estimate_order=noise_estimate_order, pvals=pvals, threshold=threshold,
                     sigterm_idx=sigterm_idx, n_term=n_term, sig_level=sig_level)

    def find(self, Xsigned_nxp):
        return np.array([np.where((self.Xsigned_nxp == X_p).all(axis=1))[0][0] for X_p in Xsigned_nxp])

    def read_poelwijk_supp3(self):
        supp3_df = pd.read_excel(
            "/home/clarafy/waterslides/calibrating-design/poelwijk/supp_data_3.xlsx", skiprows=2, header=None)
        supp3_df.columns = ["binary_genotype", "amino_acid_sequence", "counts_input", "counts_red", "counts_blue",
                            "UNK1", "brightness_red", "brightness_blue", "UNK2", "brightness_combined"]
        supp3_df["binary_genotype"] = supp3_df["binary_genotype"].apply(lambda x: x[1:-1])
        return supp3_df

    def strarr2signedarr(self, binstrarr):
        return np.array([[2 * int(b) - 1 for b in binstr] for binstr in binstrarr])

    def get_measurements(self, x_idx: np.array, seed: int = None):
        # given indices of sequences to measure, returns noisy measurements
        np.random.seed(seed)
        return np.array([np.random.normal(loc=self.y_n[i], scale=self.noise_scale * self.se_n[i]) for i in x_idx])


class WuData(Assay):
    def __init__(self, order: int = 2, noise_scale: float = 1., load_precomputed: bool = True):
        print('Only loading assayed Wu data, not imputed data.')
        intseqs, countin_n, countout_n, yorig_n = self.read_wu_supp1()
        self.order = order
        self.yorig_n = yorig_n
        if load_precomputed:
            d = np.load('/home/clarafy/waterslides/calibrating-design/wu_order{}.npz'.format(order))
            self.X_nxp = d['X_nxp']
        else:
            print("Computing encoding up to order {}.".format(order))
            self.X_nxp = util.fourier_from_seqs(intseqs, 20, order=order) # load data

        print('{} features of up to order {}'.format(self.X_nxp.shape[1], order))

        y_n = np.log((countout_n+0.5) / (countout_n[0]+0.5)) - np.log((countin_n+0.5) / (countin_n[0]+0.5))
        self.y_n = np.array(y_n)

        # Fowler et al. estimate of SE
        se_n = np.sqrt((1/(countin_n+0.5)) + (1/(countin_n[0]+0.5))+ (1/(countout_n+0.5)) + (1/(countout_n[0]+0.5)))
        self.se_n = np.array(se_n)
        self.noise_scale = noise_scale

    def read_wu_supp1(self):
        # TODO: imputed values from supp 2, but noise model?
        df = pd.read_excel("/home/clarafy/waterslides/calibrating-design/wu/wu_supp1.xlsx")
        seqs_n, y_n = list(df['Variants']), np.array(df['Fitness'])
        countin_n, countout_n = np.array(df['Count input']), np.array(df['Count selected'])
        intseqs = [util.str2ints(seq) for seq in seqs_n]
        return intseqs, countin_n, countout_n, y_n

    def get_measurements(self, x_idx: np.array, seed: int = None):
        # TODO: noise model
        # given indices of sequences to measure, returns noisy measurements
        # np.random.seed(seed)
        # return np.array([np.random.normal(loc=self.y_n[i], scale=self.noise_scale * self.se_n[i]) for i in x_idx])
        return self.yorig_n[x_idx]

class PokusaevaData(Assay):
    alphabet_szs = [2, 2, 3, 2, 2, 3, 3, 4, 2, 4, 4]
    def __init__(self, order: int = 3, noise_scale: float = 1., load_precomputed: bool = True):
        # TODO: load pre-computed Fourier encodings
        print('No noise model for Pokusaeva data.')
        d = np.load('/home/clarafy/waterslides/FitnessSparsity/results/his3p_big_data.npy', allow_pickle=True).item()
        intseqs, yorig_n = d['seq'], np.array(d['y'])
        self.order = order
        self.yorig_n = yorig_n
        if load_precomputed:
            d = np.load('/home/clarafy/waterslides/calibrating-design/pokusaeva/pokusaeva_order{}.npz'.format(order))
            self.X_nxp = d['X_nxp']
            print(self.X_nxp.shape)
        else:
            print("Computing encoding up to order {}.".format(order))
            self.X_nxp = util.fourier_from_seqs(intseqs, self.alphabet_szs, order=order)


    def get_measurements(self, x_idx: np.array, seed: int = None):
        return self.yorig_n[x_idx]
