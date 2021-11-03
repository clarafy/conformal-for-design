import numpy as np
import scipy as sc

from tqdm import tqdm
from itertools import chain, combinations

import matplotlib.pyplot as plt

AA = '-ARNDCQEGHILKMFPSTWYV'

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
            if char:  # if 0, zero embedding
                ohecol[i, intcol[i] - 1] = 1
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

def plot_trajectory(reward_sxtxk, seeds_to_plot = None, title: str = None):
    n_seed, tmp, k = reward_sxtxk.shape
    n_round = tmp - 1
    if seeds_to_plot is None:
        seeds_to_plot = range(10)

    plt.figure(figsize=(14, 4));
    for seed in seeds_to_plot:
        label = 'max reward over batch' if seed == 0 else '__nolegend__'
        plt.plot(range(n_round + 1), np.max(reward_sxtxk[seed], axis=1), '-o', linewidth=4, alpha=0.5, label=label);

    cummax_sxt = np.zeros([n_seed, n_round + 1])
    cummax_sxt[:, 0] = np.max(reward_sxtxk[:, 0])
    for t in range(1, n_round + 1):

        cummax_sxt[:, t] = np.fmax(cummax_sxt[:, t - 1], np.max(reward_sxtxk[:, t], axis=1))
    plt.plot(range(n_round + 1), np.mean(cummax_sxt, axis=0), linewidth=4, c='k',
             label='avg. cumulative max over {} trials'.format(n_seed));
    plt.xlabel('round'); plt.ylabel('reward');
    if title is not None:
        plt.title(title);
    plt.legend()

def plot_calibration(alpha: float, reward_sxtxk, miscov_sxtxkx2, len_sxtxk, intlb_sxtxk, pred_sxtxk, s_sxtxk,
                     seeds_to_plot = None, title: str = None, interval_ylim = None, width_subplots: bool = False,
                     plot_each_intlb: bool = False, xtickint: int = 5, seeds_to_violinplot = None):
    # len_sxtxk = interval_sxtxkx2[:, :, :, 1] - interval_sxtxkx2[:, :, :, 0]
    # y_sxtxk = data.y_n[arms_sxtxk] (true E[y | x])
    # res_sxtxk = reward_sxtxk - pred_sxtxk

    if seeds_to_plot is None:
        seeds_to_plot = range(10)
    if seeds_to_violinplot is None:
        seeds_to_violinplot = range(3)

    n_seed, tmp, k, _ = miscov_sxtxkx2.shape
    n_round = tmp - 1
    t0 = 2  # first round we have data to calibrate with. t = 0 is initialization, t = 1 we have predictions

    cov_sxtxk = np.prod(1 - miscov_sxtxkx2[:, t0 :, :, :], axis=3)
    miscovhigh_sxtxk = np.mean(miscov_sxtxkx2[:, t0 :, :, 0], axis=(0, 2))
    miscovlow_sxtxk = np.mean(miscov_sxtxkx2[:, t0 :, :, 1], axis=(0, 2))
    xplot = np.arange(t0, n_round + 1)

    plt.figure(figsize=(16, 9)) if width_subplots else plt.figure(figsize=(16, 4))
    plt.subplot(221) if width_subplots else plt.subplot(121)
    plt.plot(xplot, np.mean(cov_sxtxk, axis=(0, 2)), '-o', linewidth=4, alpha=0.9, label='coverage');
    plt.plot(xplot, miscovhigh_sxtxk, '-o', linewidth=3, alpha=0.9, label='upper micoverage (y < LB)');
    plt.plot(xplot, miscovlow_sxtxk, '-o', linewidth=3, alpha=0.9, label='lower miscoverage (y > UB)');
    plt.axhline(1 - alpha, linestyle='--', color='k', linewidth=2)
    plt.axhline(alpha / 2, linestyle='--', color='gray', alpha=0.7, linewidth=2)
    plt.ylabel('probability'); plt.yticks(np.arange(0, 1.01, 0.1));
    if not width_subplots:
        plt.xlabel('round (= # labeled batches)');
    plt.legend(); plt.xticks(range(0, n_round + 1, xtickint))
    if title is not None:
        plt.title(title);


    plt.subplot(222)  if width_subplots else plt.subplot(122)

    for seed in seeds_to_plot:
        plt.plot(xplot, len_sxtxk[seed, t0 :, 0], '-', c='steelblue', linewidth=3, alpha=0.4);
    plt.plot(xplot, np.median(len_sxtxk[:, t0 :], axis=(0, 2)), '-ok', linewidth=4, alpha=0.9,
             label='median of {}'.format(n_seed));

    # naive upper bound on interval width
    intub_sxt = np.zeros([n_seed, n_round + 1])
    intub_sxt[:, 0] = np.nan
    for t in range(1, n_round + 1):
        intub_sxt[:, t] = np.quantile(reward_sxtxk[:, : t], 1 - alpha / 2, axis=(1, 2)) - \
                          np.quantile(reward_sxtxk[:, : t], alpha / 2, axis=(1, 2))
    plt.plot(range(1, n_round + 1), np.mean(intub_sxt[:, 1 :], axis=0), '--s', c='gray', linewidth=2, alpha=0.9,
             label='{}-interval of rewards so far'.format(1 - alpha));

    # lower bound from residuals from current iteration
    signedres_sxtxk = reward_sxtxk - pred_sxtxk
    low_sxt = np.quantile(signedres_sxtxk, alpha / 2, axis=2)
    high_sxt = np.quantile(signedres_sxtxk, 1 - alpha / 2, axis=2)
    plt.plot(range(1, n_round + 1), np.mean(high_sxt - low_sxt, axis=0)[1 :], '--o', c='gray',
             linewidth=2, alpha=0.9, label='{}-interval of signed residual'.format(1 - alpha));

    # lower bound from measurement noise
    if plot_each_intlb:
        for i in range(k):
            label = '{}-CI of inexplainable noise'.format(1 - alpha) if i == 0 else '__nolegend__'
            plt.plot(range(n_round + 1), np.mean(intlb_sxtxk[:, :, i], axis=0), '--', c='gray',
                     linewidth=2, alpha=0.7, label=label);
    else:
        plt.plot(range(n_round + 1), np.mean(intlb_sxtxk, axis=(0, 2)), ':o', c='gray',
                 linewidth=2, alpha=0.9, label='{}-CI of measurement noise'.format(1 - alpha));


    if not width_subplots:
        plt.xlabel('round (= # labeled batches)');
    plt.xticks(range(0, n_round + 1, xtickint))
    plt.ylabel('interval width');
    plt.legend(fontsize=10, loc='upper left');
    if interval_ylim is not None:
        plt.ylim(interval_ylim);

    if width_subplots:
        plt.subplot(223)
        for seed in seeds_to_violinplot:
            plt.violinplot(list(len_sxtxk[seed, t0 :]), range(t0, n_round + 1));
        plt.xlabel('interval width'); plt.ylabel('interval width');
        plt.xlabel('round (= # labeled batches)'); plt.xticks(range(0, n_round + 1, xtickint));
        

        # correlation between score and residual
        rho_sxt = np.zeros([n_seed, n_round + 1])
        rho_sxt[:, : t0] = np.nan
        for seed in range(n_seed):
            for t in range(t0, n_round + 1):
                # rho, _ = sc.stats.spearmanr(score_sxtxk[seed, t], np.abs(signedres_sxtxk[seed, t]))
                rho, _ = sc.stats.spearmanr(len_sxtxk[seed, t], np.abs(reward_sxtxk[seed, t] - pred_sxtxk[seed, t]))
                rho_sxt[seed, t] = rho
        plt.subplot(224)
        # plt.scatter(np.tile(np.arange(t0, n_round + 1)[None, :], [n_seed, 1]), rho_sxt[:, t0 :], s=80, alpha=0.05,
        #             label='width vs. residual');
        plt.plot(range(t0, n_round + 1), np.mean(rho_sxt[:, t0 :], axis=0), '-o', linewidth=4, label='width vs. residual')

        # correlation between score and residual
        rho2_sxt = np.zeros([n_seed, n_round + 1])
        rho2_sxt[:, : t0] = np.nan
        for seed in range(n_seed):
            for t in range(t0, n_round + 1):
                rho, _ = sc.stats.spearmanr(s_sxtxk[seed, t], np.abs(reward_sxtxk[seed, t] - pred_sxtxk[seed, t]))
                rho2_sxt[seed, t] = rho
        plt.plot(range(t0, n_round + 1), np.mean(rho2_sxt[:, t0 :], axis=0), '-o', linewidth=4,
                 label='score vs. residual')
        plt.xlabel('round (= # labeled batches)'); plt.xticks(range(0, n_round + 1, xtickint));
        plt.ylabel('Spearman correlation');
        plt.legend(); plt.ylim([-1, 1]);

def plot_calibration_v2(alpha: float, reward_sxtxk, miscov_sxtxkx2, len_sxtxk, intlb_sxtxk, pred_sxtxk, jkcov_sxtxk,
                        jklen_sxtxk, jkpcov_sxtxk, jkplen_sxtxk, interval_sxtxkx2, y_n,
                     seeds_to_plot = None, title: str = None, interval_ylim = None,
                     plot_each_intlb: bool = False, xtickint: int = 5):
    # len_sxtxk = interval_sxtxkx2[:, :, :, 1] - interval_sxtxkx2[:, :, :, 0]
    # y_sxtxk = data.y_n[arms_sxtxk] (true E[y | x])

    if seeds_to_plot is None:
        seeds_to_plot = range(10)

    n_seed, tmp, k, _ = miscov_sxtxkx2.shape
    n_round = tmp - 1
    t0 = 2  # first round we have data to calibrate with. t = 0 is initialization, t = 1 we have predictions

    cov_sxtxk = np.prod(1 - miscov_sxtxkx2[:, t0 :, :, :], axis=3)
    miscovhigh_sxtxk = np.mean(miscov_sxtxkx2[:, t0 :, :, 0], axis=(0, 2))
    miscovlow_sxtxk = np.mean(miscov_sxtxkx2[:, t0 :, :, 1], axis=(0, 2))
    xplot = np.arange(t0, n_round + 1)

    plt.figure(figsize=(16, 9))
    plt.subplot(221)
    plt.plot(xplot, np.mean(cov_sxtxk, axis=(0, 2)), '-o', linewidth=4, alpha=0.9, label='conformal coverage');
    plt.plot(xplot, miscovhigh_sxtxk, '-o', linewidth=3, alpha=0.9, label='conformal upper micoverage (y < LB)');
    plt.plot(xplot, miscovlow_sxtxk, '-o', linewidth=3, alpha=0.9, label='conformal lower miscoverage (y > UB)');
    plt.axhline(1 - alpha, linestyle='--', color='k', linewidth=2)
    plt.axhline(alpha / 2, linestyle='--', color='gray', alpha=0.7, linewidth=2)

    # jackknife baseline
    plt.plot(range(1, n_round + 1), np.mean(jkcov_sxtxk[:, 1 :], axis=(0, 2)), '-o', linewidth=4, alpha=0.9, label='JK coverage');
    plt.plot(range(1, n_round + 1), np.mean(jkpcov_sxtxk[:, 1 :], axis=(0, 2)), '-o', linewidth=4, alpha=0.9, label='JK+ coverage');
    plt.ylabel('probability'); plt.yticks(np.arange(0, 1.01, 0.1));
    plt.legend(fontsize=12); plt.xticks(range(0, n_round + 1, xtickint))
    if title is not None:
        plt.title(title);


    plt.subplot(222)
    for seed in seeds_to_plot:
        plt.plot(xplot, len_sxtxk[seed, t0 :, 0], '-', c='steelblue', linewidth=3, alpha=0.4);
    plt.plot(xplot, np.median(len_sxtxk[:, t0 :], axis=(0, 2)), '-ok', linewidth=4, alpha=0.9,
             label='conformal, median of {}'.format(n_seed));
    # jackknife baseline
    plt.plot(range(1, n_round + 1), np.median(jklen_sxtxk[:, 1 :], axis=(0, 2)) , '-o', c='crimson', linewidth=4, alpha=0.9,
             label='JK, median of {}'.format(n_seed));
    plt.plot(range(1, n_round + 1), np.median(jkplen_sxtxk[:, 1 :], axis=(0, 2)), '-o', c='tab:purple', linewidth=4, alpha=0.9,
             label='JK+, median of {}'.format(n_seed));

    # naive upper bound on interval width
    intub_sxt = np.zeros([n_seed, n_round + 1])
    intub_sxt[:, 0] = np.nan
    for t in range(1, n_round + 1):
        intub_sxt[:, t] = np.quantile(reward_sxtxk[:, : t], 1 - alpha / 2, axis=(1, 2)) - \
                          np.quantile(reward_sxtxk[:, : t], alpha / 2, axis=(1, 2))
    plt.plot(range(1, n_round + 1), np.mean(intub_sxt[:, 1 :], axis=0), '--s', c='gray', linewidth=2, alpha=0.9,
             label='{}-interval of rewards so far'.format(1 - alpha));

    # lower bound from measurement noise
    if plot_each_intlb:
        for i in range(k):
            label = '{}-CI of inexplainable noise'.format(1 - alpha) if i == 0 else '__nolegend__'
            plt.plot(range(n_round + 1), np.mean(intlb_sxtxk[:, :, i], axis=0), '--', c='gray',
                     linewidth=2, alpha=0.7, label=label);
    else:
        plt.plot(range(n_round + 1), np.mean(intlb_sxtxk, axis=(0, 2)), ':o', c='gray',
                 linewidth=2, alpha=0.9, label='{}-CI of measurement noise'.format(1 - alpha));

    plt.xticks(range(0, n_round + 1, xtickint))
    plt.ylabel('interval width');
    plt.legend(fontsize=10, loc='upper left');
    if interval_ylim is not None:
        plt.ylim(interval_ylim);

    plt.subplot(223)
    # example CIs
    # parts = plt.violinplot(y_n, positions=np.array([0]), widths=2.5, points=2000);
    # for i in range(1):
    #     yerrlow_t = pred_sxtxk[0, t0 :, i] - interval_sxtxkx2[0, t0 :, i , 0]
    #     yerrhigh_t = interval_sxtxkx2[0, t0 :, i, 1] - pred_sxtxk[0, t0 :, i]
    #     yerr = np.vstack([yerrlow_t[None, :], yerrhigh_t[None, :]])
    #     plt.errorbar(xplot + i * 0.1, pred_sxtxk[0, t0 :, i], yerr=yerr, marker='.', linestyle='', color='steelblue', ms=0);
    #     plt.errorbar(xplot + i * 0.1, reward_sxtxk[0, t0 :, i], marker='*', linestyle='', color='k', ms=10);
    # plt.xlabel('interval width'); plt.ylabel('interval width');
    # plt.xlabel('round (= # labeled batches)'); plt.xticks(range(0, n_round + 1, xtickint));

    # or, histograms of interval widths at each round
    for seed in range(1):
        plt.violinplot(list(len_sxtxk[seed, t0 :]), range(t0, n_round + 1));
    for seed in range(1):
        parts = plt.violinplot(list(jklen_sxtxk[seed, 1 :]), range(1, n_round + 1));
        for p in parts['bodies']:
            p.set_facecolor('crimson');
            p.set_edgecolor('crimson');
    for seed in range(1):
        parts = plt.violinplot(list(jkplen_sxtxk[seed, 1 :]), range(1, n_round + 1));
        for p in parts['bodies']:
            p.set_facecolor('tab:purple');
            p.set_edgecolor('tab:purple');
    plt.xlabel('interval width'); plt.ylabel('interval width');
    plt.xlabel('round (= # labeled batches)'); plt.xticks(range(0, n_round + 1, xtickint));


    # correlation between score and residual
    rho_sxtx3 = np.zeros([n_seed, n_round + 1, 3])
    rho_sxtx3[:, : t0] = np.nan
    for seed in range(n_seed):
        for t in range(1, n_round + 1):
            # rho, _ = sc.stats.spearmanr(score_sxtxk[seed, t], np.abs(signedres_sxtxk[seed, t]))
            res_k = np.abs(reward_sxtxk[seed, t] - pred_sxtxk[seed, t])
            if t > 1:
                rho, _ = sc.stats.spearmanr(len_sxtxk[seed, t], res_k)
                rho_sxtx3[seed, t, 0] = rho
            rho, _ = sc.stats.spearmanr(jklen_sxtxk[seed, t], res_k)
            rho_sxtx3[seed, t, 1] = rho
            rho, _ = sc.stats.spearmanr(jkplen_sxtxk[seed, t], res_k)
            rho_sxtx3[seed, t, 2] = rho
            print(len_sxtxk[seed, t])
            print(jklen_sxtxk[seed, t])
            print('\n')
    plt.subplot(224)
    # plt.scatter(np.tile(np.arange(t0, n_round + 1)[None, :], [n_seed, 1]), rho_sxt[:, t0 :], s=80, alpha=0.05,
    #             label='width vs. residual');
    plt.plot(range(t0, n_round + 1), np.mean(rho_sxtx3[:, t0 :, 0], axis=0), '-o', linewidth=4, label='conformal')
    plt.plot(range(1, n_round + 1), np.mean(rho_sxtx3[:, 1 :, 1], axis=0), '-o', linewidth=4, label='JK', c='crimson')
    plt.plot(range(1, n_round + 1), np.mean(rho_sxtx3[:, 1 :, 2], axis=0), '-o', linewidth=4, label='JK+', c='tab:purple')


    plt.xlabel('round (= # labeled batches)'); plt.xticks(range(0, n_round + 1, xtickint));
    plt.ylabel('Spearman correlation of width vs. residual');
    plt.legend(); plt.ylim([0, 1]);

