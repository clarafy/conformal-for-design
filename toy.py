import numpy as np
import scipy as sc

from sklearn.linear_model import RidgeCV

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import poelwijk.epistasis_data as poelwijk_data

import matplotlib.pyplot as plt

class PoelwijkData(object):
    def __init__(self, order: int, shuffle: bool = False):
        df = poelwijk_data.get_supp_data_3_df()
        self.Xorig_nxp = self.strarr2signedarr(df.binary_genotype)
        self.X_nxp = higher_order_features(self.Xorig_nxp, order=order)
        self.n, self.p = self.X_nxp.shape
        self.yorig_n = np.array(df.brightness_blue)

        # Fowler computation
        y_n = np.log((df.counts_blue + 0.5) / (df.counts_blue[0] + 0.5)) - np.log((df.counts_input + 0.5) / (df.counts_input[0] + 0.5))
        self.y_n = np.array(y_n)
        red_n = np.log((df.counts_red + 0.5) / (df.counts_red[0] + 0.5)) - np.log((df.counts_input + 0.5) / (df.counts_input[0] + 0.5))

        self.red_n = np.array(red_n)

        # compute SE
        se_n = np.sqrt((1 / (df.counts_input + 0.5)) + (1 / (df.counts_input[0] + 0.5))+ (1 / (df.counts_blue + 0.5)) + (1 / (df.counts_blue[0] + 0.5)))
        self.se_n = np.array(se_n)

    def strarr2signedarr(self, binstrarr):
        return np.array([[2 * int(b) - 1 for b in binstr] for binstr in binstrarr])

    def get_measurements(self, seq_idx, seed: int = None):
        np.random.seed(seed)
        return np.array([np.random.normal(loc=self.y_n[i], scale=self.se_n[i]) for i in seq_idx])


def higher_order_features(X_nxp, order: int = 2):
    if order == 1:
        return X_nxp
    p = X_nxp.shape[1]
    featcols = []
    if order >= 2:
        for i in range(p):
            for j in range(i + 1, p):
                featcols.append(X_nxp[:, i] * X_nxp[:, j])
    if order >= 3:
        for i in range(p):
            for j in range(i + 1, p):
                for k in range(j + 1, p):
                    featcols.append(X_nxp[:, i] * X_nxp[:, j] * X_nxp[:, k])
    if order >= 4:
        for i in range(p):
            for j in range(i + 1, p):
                for k in range(j + 1, p):
                    for l in range(k + 1, p):
                        featcols.append(X_nxp[:, i] * X_nxp[:, j] * X_nxp[:, k] * X_nxp[:, l])
    if order >= 5:
        print("Only computing features up to order 4")
    return np.hstack([X_nxp, np.array(featcols).T])

def binary2signed(X_nxp):
    Xnew_nxp = X_nxp.copy()
    Xnew_nxp[np.where(X_nxp == 0)] = -1
    return Xnew_nxp

class RidgeReg(object):
    def __init__(self, dataset_name: str, alphas = None):
        if dataset_name == 'poelwijk':
            d = np.load('/home/clarafy/waterslides/calibrating-design/poelwijk.npz')
            self.X_nxla, self.y_n = d['X_nxla'], d['ycomb_n']
        self.n = self.y_n.size
        if alphas is None:
            alphas = np.logspace(-5, 5, 11)
        self.model = RidgeCV(alphas=alphas)
        self.running_train_idx = np.array([]).astype(np.int)

    def _update_train_idx(self, train_idx):
        self.running_train_idx = np.hstack([self.running_train_idx, train_idx])

    def fit(self, order: int = 1):
        if order == 1:
            self.X_nxp = self.X_nxla.copy()
        self.model.fit(self.X_nxp[self.running_train_idx], self.y_n[self.running_train_idx])

    def fit_initial(self, n_train: int, seed: int = None, verbose: bool = False):
        np.random.seed(seed)
        train_idx = np.random.choice(self.n, size=n_train, replace=False)
        self._update_train_idx(train_idx)
        self.fit()
        if verbose:
            print("Selected alpha = {:.1f}".format(self.model.alpha_))
        return train_idx

    def select_best(self, k: int = 1):
        pred_n = self.model.predict(self.X_nxp)
        best_idx = np.argsort(-pred_n)[: k]
        return best_idx, self.X_nxp[best_idx], self.y_n[best_idx]

    def refit(self, train_idx, verbose: bool = False):
        self._update_train_idx(train_idx)
        self.fit()
        if verbose:
            print("Selected alpha after refitting = {:.1f}".format(self.model.alpha_))

    def select_and_refit(self, k: int = 1, verbose: bool = False):
        best_idx, X_nxp, y_n = self.select_best(k=k)
        self.refit(best_idx, verbose=verbose)
        return best_idx, X_nxp, y_n


# ===== string conversion utilities for Poelwijk et al. (2019) data =====

def str2arr(bitstring: str):
    return np.array([float(char) for char in bitstring])

def strarr2arr(X_nxstr: np.array):
    return np.array([str2arr(X_str) for X_str in X_nxstr])

def arr2str(X_p: np.array):
    return "".join([str(int(val)) for val in X_p])

def arr2strarr(X_nxp: np.array):
    return np.array([arr2str(X_p) for X_p in X_nxp])

# ===== plotting =====

def plot_xy(x_n: np.array, y_n: np.array, color: str = "orange", alpha: float = 0.9):
    min = np.min([np.min(x_n), np.min(y_n)])
    max = np.max([np.max(x_n), np.max(y_n)])
    plt.plot([min, max], [min, max], "--", c=color, alpha=0.9);





