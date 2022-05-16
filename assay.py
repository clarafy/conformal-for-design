from abc import ABC, abstractmethod
import time
from itertools import chain, combinations

import numpy as np
import scipy as sc
import pandas as pd

from sklearn.linear_model import LinearRegression


# ===== utilities for Walsh-Hadamard transform =====
# adapted from David H. Brookes's code (https://github.com/dhbrookes/FitnessSparsity/blob/main/src/utils.py) for:
# D. H. Brookes, A. Aghazadeh, J. Listgarten,
# On the sparsity of fitness functions and implications for learning. PNAS, 119 (2022).

def get_interactions(seq_len, order: int = None):
    """
    Returns a list of tuples of epistatic interactions for a given sequence length, up to a specified order.
    For example, get_interactions(3) returns [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)].
    This sets of the order used for regression coefficients.
    """
    if order is None:
        order = seq_len
    sites = list(range(seq_len))
    combos = chain.from_iterable(combinations(sites, o) for o in range(order + 1))
    return list(combos)

def walsh_hadamard_from_seqs(signedseq_nxl: np.array, order: int = None, normalize: bool = False):
    """
    Returns an n x array of (truncated) Walsh-Hadamard encodings of a given n x array of binary sequences.
    """
    n, seq_len = signedseq_nxl.shape
    interactions = get_interactions(seq_len, order=order)
    X_nxp = np.zeros((n, len(interactions)))
    for i, idx in enumerate(interactions):
        if len(idx) == 0:
            X_nxp[:, i] = 1
        else:
            X_nxp[:, i] = np.prod(signedseq_nxl[:, idx], axis=1)
    if normalize:
        X_nxp /= np.sqrt(np.power(2, seq_len))  # for proper WH matrix
    return X_nxp


# ===== classes for handling combinatorially complete data sets =====

class Assay(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_measurements(self, x_idx: np.array, seed: int = None):
        raise NotImplementedError

class PoelwijkData(Assay):

    def __init__(self, fitness: str, order: int = 1, noise_estimate_order: int = 7, sig_level: float = 0.01,
                 load_precomputed_noise: bool = True):
        if fitness not in ['red', 'blue']:
            raise ValueError('Unrecognized fitness name: {}'.format(fitness))

        # ===== featurize sequences as higher-order interaction terms =====

        df = self.read_poelwijk_supp3()
        self.Xsigned_nxp = self.strarr2signedarr(df.binary_genotype)  # 1/-1 encoding of sequences
        self.X_nxp = walsh_hadamard_from_seqs(self.Xsigned_nxp, order=order) # featurize including intercept

        self.n, self.p = self.X_nxp.shape
        self.order = order
        print('Using {} order-{} features'.format(self.p, order))

        if fitness == 'blue':
            self.y_n = np.array(df.brightness_blue)
        elif fitness == 'red':
            self.y_n = np.array(df.brightness_red)

        # ===== estimate per-sequence measurement noise SD =====

        if load_precomputed_noise:
            d = np.load('../data/fluorescence/{}_noise.npz'.format(fitness))
            self.se_n = d['se_n']
            print("Loading estimated measurement noise SD computed using order {} and significance level {}".format(
                d['order_est_noise'], d['sig_level']))

        else:
            t0 = time.time()

            # ===== compute Walsh-Hadamard transform, truncated to order noise_estimate_order =====
            # best linear model of complete fitness landscape using terms of up to noise_estimate_order.
            # default value of noise_estimate_order = 7 taken from Poelwijk et al. (2019), who found
            # significant epistatic interactions of up to order 7 in the complete fitness landscape (see their Fig. 2e)

            # encode all 2^13 sequences with up to noise_estimate_order terms
            X_nxp = walsh_hadamard_from_seqs(self.Xsigned_nxp, order=noise_estimate_order)
            n_feat = X_nxp.shape[1]
            print('Estimating noise using {} interaction terms up to order {}'.format(n_feat, noise_estimate_order))

            # fit linear model using all 2^13 fitness measurements
            ols = LinearRegression(fit_intercept=False)  # featurization from walsh_hadamard_from_seqs has intercept
            ols.fit(X_nxp, self.y_n)

            # determine statistically significant coefficients
            # compute t-statistics
            pred_n = ols.predict(X_nxp)
            sigmasq_hat = np.sum(np.square(self.y_n - pred_n)) / (self.n - n_feat)  # estimate of \sigma^2
            var_p = sigmasq_hat * (np.linalg.inv(np.dot(X_nxp.T, X_nxp)).diagonal())
            ts_p = ols.coef_ / np.sqrt(var_p)

            # two-sided p-values
            pvals = np.array([2 * (1 - sc.stats.t.cdf(np.abs(t), (self.n - n_feat))) for t in ts_p])
            self.coef_pvals = pvals

            # use Bonferroni-Sidak correction as in Poelwijk et al. (2019) (Fig. 2e, S6)
            threshold = 1 - np.power(1 - sig_level, 1 / n_feat)
            sigterm_idx = np.where(pvals < threshold)[0]
            print("{} terms below {} for significance level {}. {:.1f} s".format(
                sigterm_idx.size, threshold, sig_level, time.time() - t0))

            # estimate per-sequence measurement noise SD by taking difference between measurements and
            # predictions made using the statistically significant coefficients
            pred_n = X_nxp[:, sigterm_idx].dot(ols.coef_[sigterm_idx])
            self.se_n = np.abs(pred_n - self.y_n)
            np.savez('../data/fluorescence/{}_noise.npz'.format(fitness),
                     se_n=self.se_n, noise_estimate_order=noise_estimate_order, pvals=pvals, threshold=threshold,
                     sigterm_idx=sigterm_idx, n_feat=n_feat, sig_level=sig_level)

    def find(self, Xsigned_nxp):
        return np.array([np.where((self.Xsigned_nxp == X_p).all(axis=1))[0][0] for X_p in Xsigned_nxp])

    def read_poelwijk_supp3(self):
        """
        Parse Poelwijk et al. (2019) Supplementary Data 3 for raw data.

        :return: pandas dataframe
        """
        df = pd.read_excel("../data/fluorescence/supp_data_3.xlsx", skiprows=2, header=None)
        df.columns = ["binary_genotype", "amino_acid_sequence", "counts_input", "counts_red", "counts_blue",
                            "UNK1", "brightness_red", "brightness_blue", "UNK2", "brightness_combined"]
        df["binary_genotype"] = df["binary_genotype"].apply(lambda x: x[1:-1])
        return df

    def strarr2signedarr(self, binstrarr):
        """
        Convert array of strings of 0s and 1s to numpy array of -1s and 1s

        :param binstrarr: iterable containing strings of 0s and 1s
        :return: numpy array where each row corresponds to string
        """
        return np.array([[2 * int(b) - 1 for b in binstr] for binstr in binstrarr])

    def get_measurements(self, seqidx_n: np.array, seed: int = None):
        """
        Given indices of sequences, return noisy measurements (using estimated measurement noise SD).

        :param seqidx_n: iterable of ints, indices of which sequences to get measurements for
        :param seed: int, random seed
        :return: numpy array of noisy measurements corresponding to provided sequence indices
        """
        np.random.seed(seed)
        noisy_n = np.array([np.random.normal(loc=self.y_n[i], scale=self.se_n[i]) for i in seqidx_n])
        # enforce non-negative measurement since enrichment scores are non-negative
        return np.fmax(noisy_n, 0)

