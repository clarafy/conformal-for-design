from abc import ABC, abstractmethod
import time
from itertools import chain, combinations

import numpy as np
import scipy as sc
import pandas as pd

from sklearn.linear_model import LinearRegression
from tensorflow.keras.utils import Sequence
from calibrate import get_invcov_dot_xt

# ===== utilities and classes for AAV experiments =====

# ----- utilities for converting between amino acids and nucleotides -----

AA2CODON = {
        'l': ['tta', 'ttg', 'ctt', 'ctc', 'cta', 'ctg'],
        's': ['tct', 'tcc', 'tca', 'tcg', 'agt', 'agc'],
        'r': ['cgt', 'cgc', 'cga', 'cgg', 'aga', 'agg'],
        'v': ['gtt', 'gtc', 'gta', 'gtg'],
        'a': ['gct', 'gcc', 'gca', 'gcg'],
        'p': ['cct', 'ccc', 'cca', 'ccg'],
        't': ['act', 'acc', 'aca', 'acg'],
        'g': ['ggt', 'ggc', 'gga', 'ggg'],
        '*': ['taa', 'tag', 'tga'],
        'i': ['att', 'atc', 'ata'],
        'y': ['tat', 'tac'],
        'f': ['ttt', 'ttc'],
        'c': ['tgt', 'tgc'],
        'h': ['cat', 'cac'],
        'q': ['caa', 'cag'],
        'n': ['aat', 'aac'],
        'k': ['aaa', 'aag'],
        'd': ['gat', 'gac'],
        'e': ['gaa', 'gag'],
        'w': ['tgg'],
        'm': ['atg']
    }


NUC_ORDERED = ['A', 'T', 'C', 'G']
NUC2IDX = {nuc: i for i, nuc in enumerate(NUC_ORDERED)}

AA_ORDERED = [k.upper() for k in AA2CODON.keys()]
AA2IDX = {aa: i for i, aa in enumerate(AA_ORDERED)}

def pnuc2paa(pnuc_Lxk):
    """
    Converts nucleotide probabilities to amino acid probabilities.
    """
    L = pnuc_Lxk.shape[0]
    paadf_kxL = pd.DataFrame(0., index=AA_ORDERED, columns=range(int(L / 3)))
    for i in range(int(L / 3)):
        for aa in AA_ORDERED:
            codons = AA2CODON[aa.lower()]
            # for each codon corresponding to the AA, compute probability of generating that codon
            for cod in codons:
                p_cod = 1
                for j in range(3): # multiply probabilities of each of the 3 nucleotides in the codon
                    nuc_idx = NUC2IDX[cod[j].upper()]
                    p_cod *= pnuc_Lxk[i * 3 + j, nuc_idx]
                paadf_kxL[i].loc[aa] += p_cod
    return np.array(paadf_kxL).T

def phinuc2paa(phinuc_Lxk):
    """
    Converts unnormalized nucleotide probabilities to amino acid probabilities.
    """
    # normalize probabilities of categorical distribution per site
    pnuc_Lxk = np.exp(phinuc_Lxk) / np.sum(np.exp(phinuc_Lxk), axis=1, keepdims=True)
    # convert nucleotide probabilities to amino acid probabilities
    paa_Lxk = np.array(pnuc2paa(pnuc_Lxk))
    return paa_Lxk

# NNK categorical distribution
pnnknucpersite = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0, 0.5, 0, 0.5]])
pnnknuc_Lxk = np.tile(pnnknucpersite, [7, 1])
PNNKAA_LXK = np.array(pnuc2paa(pnnknuc_Lxk))

# ----- rejection sampling from test distribution -----

def get_loglikelihood(seq_n, p_Lxk: np.array):
    ohe_nxLxk = np.stack([one_hot_encode(seq, flatten=False) for seq in seq_n])
    logp_1xLxk = np.log(p_Lxk)[None, :, :]
    logp_n = np.sum(ohe_nxLxk * logp_1xLxk, axis=(1, 2))
    return logp_n

def get_rejection_sampling_acceptance_probabilities(seq_n, phitestnuc_Lxk, logptrain_n):
    ptestaa_Lxk = phinuc2paa(phitestnuc_Lxk)
    ratio_Lxk = ptestaa_Lxk / PNNKAA_LXK
    maxp_l = np.max(ratio_Lxk, axis=1)
    M = np.prod(maxp_l)

    # compute test likelihoods of all data
    logptest_n = get_loglikelihood(seq_n, ptestaa_Lxk)
    paccept_n = np.exp(logptest_n - (np.log(M) + logptrain_n))
    return paccept_n, logptest_n

def rejection_sample_from_test_distribution(paccept_n):
    nonzero_samples_from_test = False
    while not nonzero_samples_from_test:
        accept_n = sc.stats.bernoulli.rvs(paccept_n)
        testsamp_idx = np.where(accept_n)[0]
        n_test = testsamp_idx.size
        if n_test:
            nonzero_samples_from_test = True
    return testsamp_idx

# ----- class for sequence-fitness data generation -----

def one_hot_encode(seq, flatten: bool = True):
    l = len(seq)
    ohe_lxk = np.zeros((l, len(AA_ORDERED)))
    ones_idx = (range(l), [AA2IDX[seq[i]] for i in range(l)])
    ohe_lxk[ones_idx] = 1
    return ohe_lxk.flatten() if flatten else ohe_lxk

class DataGenerator(Sequence):
    def __init__(self, seq_n, fitness_nx2 = None, ids = None, batch_size: int = 1000, shuffle: bool = True):
        self.seq_n = seq_n
        # (estimates of) mean and variance log enrichment score (dummy values if using for prediction)
        self.fitness_nx2 = fitness_nx2 if fitness_nx2 is not None else np.zeros([len(seq_n), 2])
        self.ids = ids if ids is not None else range(len(seq_n))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_feat = len(self.seq_n[0]) * len(AA_ORDERED)
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Update indices after each epoch.
        """
        self.idx = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        # generate indices of the batch
        idx = self.idx[index * self.batch_size : (index + 1) * self.batch_size]

        # find list of IDs
        ids = [self.ids[k] for k in idx]

        # fetch sequences and their (estimated) fitness mean and variance
        X_bxp = np.array([one_hot_encode(self.seq_n[idx], flatten=True) for idx in ids])
        y_nx2 = self.fitness_nx2[ids]
        return [X_bxp, y_nx2[:, 0], y_nx2[:, 1]]

# ===== utilities and classes for fluorescence experiments =====

# ----- utilities for Walsh-Hadamard transform -----
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

# ----- sample training and designed data according to fluorescence experiments -----

def get_training_and_designed_data(data, n, gamma, lmbda, seed: int = None):
    """
    Sample training data uniformly at random from combinatorially complete data set (Poelwijk et al. 2019),
    and sample one designed protein (w/ ground-truth label) according to design algorithm in Eq. 6 of main paper.

    :param data: assay.PoelwijkData object
    :param n: int, number of training points, {96, 192, 384} in main paper
    :param gamma: float, ridge regularization strength
    :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
    :param seed: int, random seed
    :return: numpy arrays of training sequences, training labels, designed sequence, label, and prediction
    """

    # get random training data
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(data.n, n, replace=True)
    Xtrain_nxp, ytrain_n = data.X_nxp[train_idx], data.get_measurements(train_idx)  # get noisy measurements

    # train ridge regression model
    A_pxn = get_invcov_dot_xt(Xtrain_nxp, gamma, use_lapack=True)
    beta_p = A_pxn.dot(ytrain_n)

    # construct test input distribution \tilde{p}_{X; Z_{1:n}}
    predall_n = data.X_nxp.dot(beta_p)
    punnorm_n = np.exp(lmbda * predall_n)
    Z = np.sum(punnorm_n)

    # draw test input (index of designed sequence)
    test_idx = rng.choice(data.n, 1, p=punnorm_n / Z if lmbda else None)
    Xtest_1xp = data.X_nxp[test_idx]

    # get noisy measurement and model prediction for designed sequence
    ytest_1 = data.get_measurements(test_idx)
    pred_1 = Xtest_1xp.dot(beta_p)
    return Xtrain_nxp, ytrain_n, Xtest_1xp, ytest_1, pred_1

# ----- classes for handling combinatorially complete data sets -----

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
        # enforce non-negative measurement since enrichment scores are always non-negative
        return np.fmax(noisy_n, 0)

