from unittest import TestCase

import numpy as np
import calibrate as cal
import assay

from sklearn.linear_model import Ridge

# class TestDistributions(TestCase):
#     def test_get_uniform_train_and_design_test(self):
#         seed, lmbda, n_train, gamma = 0, 10, 48, 10
#         np.random.seed(seed)
#         data = assay.PoelwijkData('blue', order=2)
#         model = Ridge(alpha=gamma, fit_intercept=True)
#
#         # get random training data
#         rng = np.random.default_rng(0)
#         train_idx1 = rng.choice(data.n, n_train, replace=True)
#         Xtrain_nxp, ytrain_n1 = data.X_nxp[train_idx1], data.get_measurements(train_idx1)
#
#         # train model (exclude intercept feature)
#         model.fit(Xtrain_nxp[:, 1 :], ytrain_n1)
#
#         # construct test covariate distribution
#         predall_n = model.predict(data.X_nxp[:, 1 :])
#         punnorm_n = np.exp(lmbda * predall_n)
#         Z = np.sum(punnorm_n)
#
#         # draw test covariate
#         test_idx1 = rng.choice(data.n, 1, p=punnorm_n / Z)
#         ytest_nx1 = data.get_measurements(test_idx1)
#         pred_11 = model.predict(data.X_nxp[test_idx1, 1 :])
#
#         # from function
#         _, ytrain_n2, _, ytest_n2, pred_12 = cal.get_uniform_train_and_design_test(
#             data, n_train, gamma, lmbda, seed=seed)
#         # self.assertTrue(np.all(ytrain_n1 == ytrain_n2))
#         # self.assertTrue(np.all(ytest_nx1 == ytest_n2))
#         # HERE: test no longer consistent w/o setting seeds
#         print(np.abs(pred_11 - pred_12))
#         self.assertTrue(np.abs(pred_11 - pred_12) < 7.78e-16)


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

    def test_get_insample_adaptive_scores(self):
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
            scores1_n1 = cal.get_scores(model, Xaug_n1xp[:, 1 :], yaug_n1,
                                        use_loo_score=False, use_adaptive_score=True)
            scores1_n1xy[:, y_idx] = scores1_n1

        # test ridge-specific computation
        ridgeconf = cal.ConformalRidgeCovariateIntervention(None, ys, Xuniv_uxp, gamma)
        scores2_n1xy = ridgeconf.get_insample_scores(Xaug_n1xp, ytrain_n, use_adaptive_score=True)
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

    def test_lrs_equal_one(self):
        n, p, gamma, lmbda = 384, 92, 100, 1
        ys = np.arange(-1, 2, 0.1)
        ptrain_fn = lambda x: (1.0 / (n + 1)) * np.ones([x.shape[0]])

        np.random.seed(0)
        Xaug_n1xp = np.hstack([np.ones([n + 1, 1]), np.random.randn(n + 1, p)])
        ytrain_n = np.random.randn(n)
        Xuniv_uxp = Xaug_n1xp.copy()

        ridgeconf = cal.ConformalRidgeCovariateIntervention(ptrain_fn, ys, Xuniv_uxp, gamma)
        _, w_n1xy = ridgeconf.get_loo_scores_and_lrs(
            Xaug_n1xp, ytrain_n, 0, use_adaptive_score=False)
        # check p_train = p_test
        self.assertTrue(np.max(np.abs(w_n1xy - 1)) < 1e-16)


    def test_get_loo_adaptive_scores(self):
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
            scores1_n1 = cal.get_scores(model, Xaug_n1xp[:, 1 :], yaug_n1,
                                        use_loo_score=True, use_adaptive_score=True)
            scores1_n1xy[:, y_idx] = scores1_n1

        # test ridge-specific computation
        ridgeconf = cal.ConformalRidgeCovariateIntervention(ptrain_fn, ys, Xuniv_uxp, gamma)
        scores2_n1xy, w2_n1xy = ridgeconf.get_loo_scores_and_lrs(
            Xaug_n1xp, ytrain_n, lmbda, use_adaptive_score=True)

        # compare
        self.assertTrue(np.max(np.abs(scores1_n1xy - scores2_n1xy)) < 1.34e-15)

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

    def test_lrs_equal_one(self):
        n, p, gamma, lmbda = 384, 92, 100, 1
        ys = np.arange(-1, 2, 0.1)
        ptrain_fn = lambda x: (1.0 / (n + 1)) * np.ones([x.shape[0]])

        np.random.seed(0)
        Xaug_n1xp = np.hstack([np.ones([n + 1, 1]), np.random.randn(n + 1, p)])
        ytrain_n = np.random.randn(n)
        Xuniv_uxp = Xaug_n1xp.copy()

        ridgeconf = cal.ConformalRidgeCovariateShift(ptrain_fn, ys, Xuniv_uxp, gamma)
        _, w_n1xy = ridgeconf.get_loo_scores_and_lrs(
            Xaug_n1xp, ytrain_n, 0, use_adaptive_score=False)
        # check p_train = p_test
        print(np.max(np.abs(w_n1xy - 1)))
        self.assertTrue(np.max(np.abs(w_n1xy - 1)) < 1e-16)
