import numpy as np
import scipy as sc

from tqdm import tqdm
from itertools import chain, combinations

import matplotlib.pyplot as plt

AA = '-ARNDCQEGHILKMFPSTWYV'

def plot_xy(x_n: np.array, y_n: np.array, color: str = "orange", alpha: float = 0.9):
    min = np.min([np.min(x_n), np.min(y_n)])
    max = np.max([np.max(x_n), np.max(y_n)])
    plt.plot([min, max], [min, max], "--", c=color, alpha=0.9);


def plot_predictions(y_n, pred_n, title: str = None, **kwargs):
    plt.scatter(y_n, pred_n, **kwargs)
    plt.xlabel('fitness'); plt.ylabel('prediction');
    plt.title(title)

def train_and_evaluate(n_train, model, X_nxp, y_n, n_trial: int = 100, plot_representative: bool = False,
                      title: str = None, record_reg: bool = True, plot_xy_line: bool = False, **kwargs):
    metrics_sx3 = np.zeros([n_trial, 3])
    pred_sxn = np.zeros([n_trial, y_n.size - n_train])
    reg_s = np.nan * np.ones([n_trial])
    for seed in range(n_trial):
        np.random.seed(seed)
        shuffle_idx = np.random.permutation(y_n.size)
        train_idx, test_idx = shuffle_idx[: n_train], shuffle_idx[n_train :]
        model.fit(X_nxp[train_idx], y_n[train_idx])
        pred_n, rmse, rho, p_rho, r, p_r = evaluate(model, X_nxp[test_idx], y_n[test_idx], plot=False)
        pred_sxn[seed] = pred_n
        metrics_sx3[seed] = rmse, rho, r
        if record_reg:
            reg_s[seed] = model.alpha_

    if plot_representative:
        plot_seed = np.argmin(np.abs(metrics_sx3[:, 0] - np.mean(metrics_sx3[:, 0])))
        np.random.seed(plot_seed)
        test_idx = np.random.permutation(y_n.size)[n_train :]
        tmp = '{}\n'.format(title) if title is not None else ''
        title = '{}RMSE {:.3f}, Spearman {:.3f}, Pearson {:.3f}'.format(tmp, *np.mean(metrics_sx3, axis=0))
        plot_predictions(y_n[test_idx], pred_sxn[plot_seed], title, **kwargs)
        if plot_xy_line:
            plot_xy(y_n[test_idx], pred_sxn[plot_seed])
            plt.ylim((0.1, 0.3))
    return metrics_sx3, reg_s


def evaluate(model, X_nxp, y_n, plot: bool = False, title: str = None, **kwargs):
    pred_n = model.predict(X_nxp)
    rmse = np.sqrt(np.mean(np.square(pred_n - y_n)))
    rho, p_rho = sc.stats.spearmanr(y_n, pred_n)
    r, p_r = sc.stats.pearsonr(y_n, pred_n)
    if plot:
        tmp = '{}\n'.format(title) if title is not None else ''
        title = '{}RMSE {:.3f}, Spearman {:.3f}, Pearson {:.3f}'.format(tmp, rmse, rho, r)
        plot_predictions(y_n, pred_n, title, **kwargs)
    return pred_n, rmse, rho, p_rho, r, p_r

# ===== utilities for WHT and FT, adapted from David's code =====

def get_interactions(seq_len, order: int = None):
    """
    Returns a list of tuples of epistatic interactions for a given sequence length, up to a specified order.
    For example, get_all_interactions(3) returns [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)].
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
    This will return the same array as fourier_from_seqs(bin_seqs, 2), but is much faster.
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


def complete_graph_evs(q):
    """
    Returns eigenvectors of complete graph of size q as column vectors of a matrix
    """
    x = np.ones(q)
    y = np.eye(q)[0]
    v = x - np.linalg.norm(x, ord=2) * y
    w = v / np.linalg.norm(v, ord=2)
    w = w.reshape(q, 1)
    P = np.eye(q) - 2*np.dot(w, w.T)
    return P

def get_encodings(qs):
    """
    Returns a length-L list of arrays containing the encoding vectors corresponding
    to each character at each position in sequence, given the alphabet size at each position.
    """
    encodings = []
    Pqs = []
    L = len(qs)
    for i in range(L):
        qi = qs[i]
        Pq = complete_graph_evs(qi) * np.sqrt(qi)
        Pqs.append(Pq)
        enc_i = Pq[:, 1 :]
        encodings.append(enc_i)
    return encodings


def fourier_for_seq(intseq, encodings, order):
    """
    Returns an M x 1 array containing the Fourier encoding of a sequence,
    given the integer representation of the sequence and the encodings returned
    by get_encodings, where M = prod(qs) and qs is the alphabet size at each position.
    """
    seq_len = len(intseq)
    interactions = get_interactions(seq_len, order=order)
    interactions = [list(idx) for idx in interactions]
    epistatic_encodings = []

    enc_1 = encodings[0][intseq[0]]  # encoding for the character at site 1
    # initialize all kronecker products with either the site 1 encoding,
    # or 1 if the epistatic interaction does not involve site 1
    for idx in interactions:
        if len(idx) > 0 and 0 == idx[0]:
            enc = enc_1
            idx.pop(0)  # peel off to expose remaining sites in this interaction term (for next for-loop)
        else:
            enc = np.array([1])  # constant term
        epistatic_encodings.append(enc)

    # iterate through all remaining sites
    for l in range(1, seq_len):
        enc_l = encodings[l][intseq[l]]  # encoding for the character at site l
        for which_interaction, idx in enumerate(interactions):
            enc = epistatic_encodings[which_interaction]  # current kronecker product for this interaction
            if len(idx) > 0 and l == idx[0]:
                # if this interaction involves site l, take product with site l encoding
                enc = np.kron(enc, enc_l)
                idx.pop(0)  # peel off to expose remaining sites in this interaction term
            epistatic_encodings[which_interaction] = enc
    return np.concatenate(epistatic_encodings)


def fourier_from_seqs(intseqs, alphabet_szs, order: int = None, normalize: bool = False):
    """
    Returns an n x p array containing the Fourier encodings of a given list of
    N sequences with alphabet sizes qs.
    """
    n, seq_len = len(intseqs), len(intseqs[0])
    if order is None:
        order = seq_len
    if type(alphabet_szs) == int:
        alphabet_szs = seq_len * [alphabet_szs]

    encodings = get_encodings(alphabet_szs)  # seq_len-length list of alphabet_sz x (alphabet_sz - 1) encoding array
    X_nxp = []  # TODO: useful to pre-allocate to check memory, but need to calculate n_feat for variable alphabet_sz
    for seq in tqdm(intseqs):
        enc = fourier_for_seq(seq, encodings, order=order)
        X_nxp.append(enc.T)
    X_nxp = np.vstack(X_nxp)
    if normalize:
        X_nxp /= np.sqrt(np.prod(alphabet_szs))
    return X_nxp


# ==== representation conversion =====

def str2ints(aastr):
    return [AA.index(aa.upper()) for aa in aastr]

def intarr2ohearr(Xint_nxp, alphasz_p):
    def intcol2ohe(intcol, alpha_sz):
        ohecol = np.zeros([intcol.size, alpha_sz - 1])
        for i in range(intcol.size):
            char = intcol[i]
            if char > 1:
                ohecol[i, char - 2] = 1
        return ohecol

    X_nxp = []
    for i in range(Xint_nxp.shape[1]):
        X_nxp.append(intcol2ohe(Xint_nxp[:, i], alphasz_p[i]))
    return np.hstack(X_nxp)


# ===== plotting =====

def plot_xy(x_n: np.array, y_n: np.array, color: str = "orange", alpha: float = 0.9):
    min = np.min([np.min(x_n), np.min(y_n)])
    max = np.max([np.max(x_n), np.max(y_n)])
    plt.plot([min, max], [min, max], "--", c=color, alpha=0.9);


