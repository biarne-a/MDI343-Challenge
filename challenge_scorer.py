import numpy as np
from sklearn.metrics.scorer import _BaseScorer
from mix_matrix_factory import MixMatrixFactory


class ChallengeScorer(_BaseScorer):
    def __init__(self, combination_indices):
        super().__init__(None, -1, None)
        self.look_at_FAR_ = 0.0001
        self.combination_indices_ = combination_indices
        self.mix_matrix_factory_ = MixMatrixFactory(combination_indices)

    def compute_eval(self, fused_score):
        # calculating FAR and FRR
        sort = np.argsort(fused_score[:, 1])
        scores = fused_score[sort]
        totpos = sum(scores[:, 0])
        totneg = scores.shape[0] - totpos
        fa = (np.cumsum(scores[:, 0] - 1) + totneg) / totneg
        fr = np.cumsum(scores[:, 0]) / totpos

        i = 0
        while fa[i] > self.look_at_FAR_:
            i += 1

        # We multiply by -1 because lower is better
        return -1 * fr[i]

    def score(self, X, y, mix_matrix):
        X = np.hstack((np.ones(len(X)).reshape(-1, 1), X))

        size = len(X) // 4
        bins = [(0, size), (size + 1, 2 * size), (2 * size + 1, 3 * size), (3 * size + 1, len(X))]
        all_fuse = np.zeros((len(X), 2))
        for bin in bins:
            start = bin[0]
            end = bin[1]
            scores_matrix = X[start:end, None, :] * X[start:end, :, None]
            scores = np.multiply(scores_matrix, mix_matrix)
            y_bin = np.reshape(y[start:end], [-1, 1])
            fuse = np.sum(scores, axis=(1, 2))
            fuse = np.reshape(fuse, [-1, 1])
            fuse = np.concatenate([y_bin, fuse], axis=1)
            fuse[np.isnan(fuse)] = -float("inf")
            all_fuse[start:end] = fuse

        return self.compute_eval(all_fuse)

    def __call__(self, clf, X, y, sample_weight=None):
        super().__call__(clf, X, y, sample_weight=sample_weight)

        if len(clf.coef_) > 1:
            raise AssertionError("The coef size should be 1")

        print(clf.coef_[0])

        mix_matrix = self.mix_matrix_factory_.create_partial_mix_matrix(clf.coef_[0])

        # We should only keep the first order features because the 2 second order features
        # Will be included using the mix matrix
        X = X[:, :len(self.combination_indices_)]

        # Cross validation of LogisticRegression from sklearn replaces the 0 y value by -1
        # We have to inverse that for our score function to work
        y[y == -1] = 0

        return self.score(X, y, mix_matrix)
