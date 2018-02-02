import numpy as np


class MixMatrixFactory:
    def __init__(self, combination_indices):
        self.combination_indices_ = combination_indices
        self.nb_algo_ = len(combination_indices)

    def create_partial_mix_matrix(self, coefs):
        indices_x, indices_y = np.triu_indices(self.nb_algo_)
        indices_x = np.array(indices_x) + 1
        indices_y = np.array(indices_y) + 1
        indices = (tuple(indices_x), tuple(indices_y))
        result = np.zeros((self.nb_algo_ + 1, self.nb_algo_ + 1))
        result[0, 1:] = coefs[:self.nb_algo_]
        result[indices] = coefs[self.nb_algo_:]
        return result

    def create_full_mix_matrix(self, coefs):
        indices = np.triu_indices(self.nb_algo_)
        coefs_matrix = np.zeros((self.nb_algo_, self.nb_algo_))
        coefs_matrix[indices] = coefs[self.nb_algo_:]
        result = np.zeros((15, 15))
        combination_indices = np.array(self.combination_indices_) + 1
        result[0, combination_indices] = coefs[:self.nb_algo_]
        for i in range(len(combination_indices)):
            index_i = combination_indices[i]
            result[index_i, combination_indices] = coefs_matrix[i, :]
        return result


# matrix_factory = MixMatrixFactory([1, 2])
# mix_matrix = matrix_factory.create_partial_mix_matrix([0.1, 0.2, 0.01, 0.02, 0.04])
#
# matrix_factory = MixMatrixFactory([0, 2, 4])
# mix_matrix = matrix_factory.create_full_mix_matrix([0.1, 0.3, 0.5, 0.01, 0.03, 0.05, 0.09, 0.15, 0.25])
#
# pass