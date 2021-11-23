from unittest import TestCase

import numpy as np
import calibrate as cal

from sklearn.linear_model import Ridge


class TestWeightedQuantile(TestCase):
    def test_weighted_quantile(self):
        vals_n = np.array([2, 1, 3])
        weights_n = np.array([0.1, 0.5, 0.4])
        quantile = 0.2
        q = cal.weighted_quantile(vals_n, weights_n, quantile)
        self.assertTrue(q == 1)

    def test_get_quantile(self):
        scores_nxy = np.array([2, 1, 3])[:, None]
        w_nxy = np.array([0.1, 0.5, 0.4])[:, None]
        q = cal.get_quantile(0.8, w_nxy, scores_nxy)
        self.assertTrue(q == 1)

    def test_get_quantile_dimensions(self):
        scores_nxy = np.array([2, 1, 3])
        w_nxy = np.array([0.1, 0.5, 0.4])
        q = cal.get_quantile(0.8, w_nxy, scores_nxy)
        self.assertTrue(q == 1)

    def test_get_quantile_inf(self):
        scores_nxy = np.array([2, 1, 3])[:, None]
        w_nxy = np.array([0.1, 0.5, 0.4])[:, None]
        q = cal.get_quantile(0.2, w_nxy, scores_nxy)
        self.assertTrue(q == np.inf)

class TestConformalRidgeCovariateIntervention(TestCase):

    def test_get_insample_scores(self):
        n, p, gamma = 24, 92, 100
        ys = np.arange(-1, 2, 0.01)

        np.random.seed(0)
        Xaug_n1xp = np.hstack([np.ones([n + 1, 1]), np.random.randn(n + 1, p)])
        ytrain_n = np.random.randn(n)
        Xuniv_uxp = Xaug_n1xp.copy()

        # naive model-agnostic computation
        model = Ridge(alpha=gamma, fit_intercept=True)
        scores1_n1xy = np.zeros([n + 1, ys.size])
        for y_idx, y in enumerate(ys):
            yaug_n1 = np.hstack([ytrain_n, y])
            scores1_n1 = cal.get_scores(model, Xaug_n1xp[:, 1 :], yaug_n1, use_loo_score=False)
            scores1_n1xy[:, y_idx] = scores1_n1

        # test ridge-specific computation
        ridgeconf = cal.ConformalRidgeCovariateIntervention(None, ys, Xuniv_uxp, gamma)
        scores2_n1xy = ridgeconf.get_insample_scores(Xaug_n1xp, ytrain_n)
        self.assertTrue(np.max(np.abs(scores1_n1xy - scores2_n1xy)) < 1.12e-15)

    def test_get_loo_scores_and_lrs(self):
        n, p, gamma, lmbda = 24, 92, 100, 10
        ys = np.arange(-1, 2, 0.1)
        # something that is invariant to intercept presence or absence
        ptrain_fn = lambda x: x[:, -1] + 1

        np.random.seed(0)
        Xaug_n1xp = np.hstack([np.ones([n + 1, 1]), np.random.randn(n + 1, p)])
        ytrain_n = np.random.randn(n)
        Xuniv_uxp = Xaug_n1xp.copy()

        # naive model-agnostic computation for scores
        model = Ridge(alpha=gamma, fit_intercept=True)
        scores1_n1xy = np.zeros([n + 1, ys.size])
        for y_idx, y in enumerate(ys):
            yaug_n1 = np.hstack([ytrain_n, y])
            scores1_n1 = cal.get_scores(model, Xaug_n1xp[:, 1 :], yaug_n1, use_loo_score=True)
            scores1_n1xy[:, y_idx] = scores1_n1

        # naive model-agnostic computation for likelihood ratios
        w1_n1xy = np.zeros([n + 1, ys.size])
        naiveconf = cal.ConformalCovariateIntervention(model, ptrain_fn, ys, Xuniv_uxp[:, 1 :])
        for y_idx, y in enumerate(ys):
            yaug_n1 = np.hstack([ytrain_n, y])
            w_n1 = naiveconf.get_lrs(Xaug_n1xp[:, 1 :], yaug_n1, lmbda)
            w1_n1xy[:, y_idx] = w_n1

        # test ridge-specific computation
        ridgeconf = cal.ConformalRidgeCovariateIntervention(ptrain_fn, ys, Xuniv_uxp, gamma)
        scores2_n1xy, w2_n1xy = ridgeconf.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # compare
        self.assertTrue(np.max(np.abs(scores1_n1xy - scores2_n1xy)) < 5.56e-16)
        self.assertTrue(np.max(np.abs(w1_n1xy - w2_n1xy)) < 7.11e-15)

class TestConformalRidgeCovariateShift(TestCase):

    def test_get_lrs(self):
        n, p, gamma, lmbda = 24, 92, 100, 6
        ys = np.arange(-1, 2, 0.01)
        # something that is invariant to intercept presence or absence
        ptrain_fn = lambda x: x[:, -1] + 1

        np.random.seed(0)
        Xaug_n1xp = np.hstack([np.ones([n + 1, 1]), np.random.randn(n + 1, p)])
        ytrain_n = np.random.randn(n)
        yaug_n1 = np.hstack([ytrain_n, np.inf])  # dummy variable
        Xuniv_uxp = Xaug_n1xp.copy()

        # naive model-agnostic computation
        model = Ridge(alpha=gamma, fit_intercept=True)
        naiveconf = cal.ConformalCovariateShift(model, ptrain_fn, ys, Xuniv_uxp[:, 1 :])
        w1_n1 = naiveconf.get_lrs(Xaug_n1xp[:, 1 :], yaug_n1, lmbda)

        # test ridge-specific computation
        ridgeconf = cal.ConformalRidgeCovariateShift(ptrain_fn, ys, Xuniv_uxp, gamma)
        w2_n1 = ridgeconf.get_lrs(Xaug_n1xp, ytrain_n, lmbda)
        self.assertTrue(np.max(np.abs(w1_n1 - w2_n1)) < 7.78e-16)
