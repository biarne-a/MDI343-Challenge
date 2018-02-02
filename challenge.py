import numpy as np
import itertools
import pickle
from mix_matrix_factory import MixMatrixFactory
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from challenge_scorer import ChallengeScorer


# Running time of each algorithm (in milliseconds)
alg_times = np.zeros((14, 1))
alg_times[0] = 163
alg_times[1] = 163
alg_times[2] = 190
alg_times[3] = 190
alg_times[4] = 206
alg_times[5] = 206
alg_times[6] = 120
alg_times[7] = 120
alg_times[8] = 83
alg_times[9] = 83
alg_times[10] = 83
alg_times[11] = 83
alg_times[12] = 170
alg_times[13] = 170

# Time constraint: The total duration of the algorithms cannot exceed 600 milliseconds
alg_time_thr = 600


##############################################################################
# Save / load data

def load_data(train_fname):
    train_data = np.loadtxt(train_fname, dtype=np.float)
    train_data_cleaned = train_data[np.all(~np.isinf(train_data), axis=1), :]
    y_trn = train_data_cleaned[:, 0].astype(int)
    X_trn = train_data_cleaned.copy()
    # we skip the first column which contains the y
    X_trn = X_trn[:, 1:]
    return X_trn, y_trn


def save_to_pickle():
    X_trn, y_trn = load_data('data/train15_telecom.txt')
    with open("data/X_trn.pickle", "wb") as f:
        pickle.dump(X_trn, f)
    with open("data/y_trn.pickle", "wb") as f:
        pickle.dump(y_trn, f)


def load_binary_data():
    with open("data/X_trn.pickle", "rb") as f:
        X_trn = pickle.load(f)
    with open("data/y_trn.pickle", "rb") as f:
        y_trn = pickle.load(f)
    return X_trn, y_trn



##############################################################################
# test

# Compute the total computational time for the fusion algorithm
def compute_total_time(M):
    is_used = np.zeros((14, 1))
    for i in range(15):
        for j in range(15):
            if (M[i, j] != 0):
                if (i >= 1):
                    is_used[i - 1] = 1
                if (j >= 1):
                    is_used[j - 1] = 1

    total_dur = np.dot(is_used.T, alg_times)
    return total_dur[0, 0]


##############################################################################
# Compute different combinations

def compute_algo_combinations(algo_times, max_budget, nb_indexes):
    np_algo_times = np.array(algo_times)
    indexes = range(14)
    combinations = []
    for combi in itertools.combinations(indexes, nb_indexes):
        combi = list(combi)
        total_time = np_algo_times[combi].sum()
        if total_time <= max_budget:
            combinations.append(combi)
    return combinations



def add_quadratic_features(X_trn, combi_train, indexes):
    tri_indices = np.triu_indices(len(indexes))
    features = np.zeros((len(X_trn), len(tri_indices[0])))
    nb_feature = 0
    for i in range(len(indexes)):
        first_feature_index = indexes[i]
        for j in range(i, len(indexes)):
            second_feature_index = indexes[j]
            feature = X_trn[:, first_feature_index] * X_trn[:, second_feature_index]
            features[:, nb_feature] = feature
            nb_feature += 1
    return np.hstack((combi_train, features))


def try_keep_best_frr(best_frrs, frr, M):
    last_frr = best_frrs[-1][0]
    if frr > last_frr:
        return False
    for i in reversed(range(len(best_frrs))):
        if i == 0 or best_frrs[i - 1][0] < frr:
            break
        best_frrs[i] = best_frrs[i - 1]
    best_frrs[i] = (frr, M)
    return True


def get_X_from_combination(X_trn, combination_indices):
    combi_train = X_trn[:, combination_indices]
    return add_quadratic_features(X_trn, combi_train, combination_indices)


def test_combination(combination, X_trn, y_trn, with_cv):
    print("Testing combination " + str(combination))

    combination_indices = np.array(combination)
    X_trn_polynomial = get_X_from_combination(X_trn, combination_indices)

    scorer = ChallengeScorer(combination_indices)
    if with_cv:
        clf = LogisticRegressionCV(cv=10, Cs=10, refit=False, scoring=scorer, solver='liblinear')
    else:
        clf = LogisticRegression()
    clf.fit(X_trn_polynomial, y_trn)

    mix_matrix_factory = MixMatrixFactory(combination_indices)
    mix_matrix = mix_matrix_factory.create_full_mix_matrix(clf.coef_[0])

    frr = scorer.score(X_trn, y_trn, mix_matrix)
    print("frr = " + str(frr))
    print("coef = " + str(clf.coef_[0]))

    return mix_matrix


def compute_best_frr(nb_algo, X_trn, y_trn):
    algo_combinations = compute_algo_combinations(alg_times, 600, nb_algo)
    print("Total number of combinations with " + str(nb_algo) +
          " combinations : " + str(len(algo_combinations)))
    min_frrs = np.array([(float("inf"), None)] * 10)
    for combi in algo_combinations:
        frr, M = test_combination(combi, X_trn, y_trn)
        try_keep_best_frr(min_frrs, frr, M)
    return min_frrs


def save_best_combinations():
    X_trn, y_trn = load_binary_data()
    min_frrs4 = compute_best_frr(4, X_trn, y_trn)
    min_frrs3 = compute_best_frr(3, X_trn, y_trn)
    min_frrs = np.vstack((min_frrs3, min_frrs4))

    for i, min_frr_M in enumerate(min_frrs):
        min_frr, M = min_frr_M
        print("best frr = " + str(min_frr))
        np.savetxt('M_pred_' + str(i) + '.txt', M, fmt='%f')


def save_combinations(combinations, with_cv):
    X_trn, y_trn = load_binary_data()
    for i, combination in enumerate(combinations):
        mix_matrix = test_combination(combination, X_trn, y_trn, with_cv)
        np.savetxt('M_pred_' + str(i) + '.txt', mix_matrix, fmt='%10.20f')


def build_combination_from_previous_run(previous_run):
    result = []
    for i in range(20):
        M = np.loadtxt(previous_run + "/M_pred_" + str(i) + ".txt", dtype=np.float)
        M_first_line = M[0, :].reshape(-1, 1)
        non_zeros_indices = np.where(np.any(M_first_line, axis=1))[0] - 1
        result.append(non_zeros_indices)
    return result


# combinations = build_combination_from_previous_run("Predictions_31_01")
save_combinations([
[7 8 11 12],
[7 8 11 13],
[6 8 11 13],
[2 8 10 13],
[7 8 10 13],
[8 11 12],
[8 10 13],
[7 8 11],
[8 11 13],
[6 8 11]
], True)
# save_combinations([[0, 8, 10, 12]])

