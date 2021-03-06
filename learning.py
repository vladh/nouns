import time
import itertools
from collections import defaultdict
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
# import pyphen
import data


GRAPHS_DIR = 'graphs/'
ESTIMATED_RUNTIME_S = 3
FREQSET_MIN_MAGNITUDE = 700
FREQSET_MIN_PROPORTION = 0.1
FREQSET_MAGNITUDE_FACTOR = 100000
N_FREQSETS = 100
N_FREQWORDS = False
N_FOLDS = 20
# dic = pyphen.Pyphen(lang='de_DE')


def get_syllable(nouninfo_item):
    return nouninfo_item[2]


def get_gender(nouninfo_item):
    return nouninfo_item[1]


def get_last_syllable(nouninfo_item):
    return get_syllable(nouninfo_item).split('-')[-1]


def get_genders_and_last_syllables(nouninfo):
    """
    For a nouninfo list, returns:

    [['f', 'chen'], ...]
    """
    return [[get_gender(d), get_last_syllable(d)] for d in nouninfo]


def get_value_map(values):
    """
    Returns [do_map, do_map_inv, unique_values], where:
        * do_map maps the given values to a series of integers,
        * do_map_inv does the reverse,
        * unique_values is the input data but unique.
    """
    unique_values = np.unique(values)
    value_map = dict(zip(unique_values, range(0, len(unique_values))))
    value_map_inv = dict(zip(range(0, len(unique_values)), unique_values))

    def do_map(d):
        return value_map[d]

    def do_map_inv(d):
        return value_map_inv[d]

    return [do_map, do_map_inv, unique_values]


def make_features_and_labels(nouninfo, gender_to_id, syllable_to_id):
    """
    Returns [features, labels], where features are:

        [[syllable_id], [...], ...]

    and labels are:

        [gender_id, ...]

    Both features and labels are np.arrays.
    """
    features = [
        [syllable_to_id(get_last_syllable(d))]
        for d in nouninfo
    ]
    labels = [
        gender_to_id(get_gender(d))
        for d in nouninfo
    ]
    return [np.array(features), np.array(labels)]


def run_test(train_features, train_labels, test_features, test_labels):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    pred_comparison = pred == test_labels
    stats = {
        'accuracy': np.sum(pred_comparison == True) / len(test_labels), # noqa
    }
    return [stats, clf]


def export_graph(clf, class_names):
    with open(f'{GRAPHS_DIR}tree-{str(time.time())}.dot', 'w') as fp:
        tree.export_graphviz(
          clf, out_file=fp, class_names=class_names, filled=True, rounded=True,
          special_characters=True, max_depth=10,
        )


def run_crossvalidation(
    genders_and_syllables,
    gender_to_id, id_to_gender, unique_genders,
    syllable_to_id, id_to_syllable, unique_syllables,
    all_features, all_labels
):
    print(f'>>> {len(all_features)} nouns (samples)')
    for gender in unique_genders:
        print(f'    {np.sum(all_labels == gender_to_id(gender))} {gender}')

    print(f'>>> Running {N_FOLDS} splits')
    kfold = KFold(n_splits=N_FOLDS)
    kfold.get_n_splits(all_features)

    accuracies = []
    for idx, [train_index, test_index] in enumerate(kfold.split(all_features)):
        train_features = all_features[train_index]
        test_features = all_features[test_index]
        train_labels = all_labels[train_index]
        test_labels = all_labels[test_index]
        [stats, clf] = run_test(
            train_features, train_labels,
            test_features, test_labels,
        )
        accuracies.append(stats['accuracy'])
        print(f'> Fold {idx}, accuracy = {round(stats["accuracy"] * 100, 2)}%')
        # if idx == 0:
        #     print('Exporting graph...')
        #     export_graph(clf, unique_genders)
        if idx > 20:
            break
    avg_accuracy = np.mean(accuracies)
    print(f'> Average accuracy: {round(avg_accuracy * 100, 2)}%')
    print()
    return avg_accuracy


def get_freqset_proportion(freqset):
    freqset_values = list(freqset[1].values())
    freqset_sum = np.sum(freqset_values)
    freqset_max = np.max(freqset_values)
    return freqset_max / freqset_sum


def get_freqset_magnitude(freqset):
    return np.sum(list(freqset[1].values()))


def get_freqset_relevance(freqset):
    freqset_proportion = get_freqset_proportion(freqset)
    freqset_magnitude = get_freqset_magnitude(freqset)
    if freqset_magnitude < FREQSET_MIN_MAGNITUDE:
        return 0
    if freqset_proportion < FREQSET_MIN_PROPORTION:
        return 0
    return (freqset_magnitude / FREQSET_MAGNITUDE_FACTOR) * freqset_proportion


def is_nice_freqset(freqset):
    if len(freqset[0]) == 0:
        return False
    if get_freqset_relevance(freqset) == 0:
        return False
    return True


def print_freqsets(freqsets):
    print(
        f'{len(freqsets)} syllables with ' +
        f'(freq > {FREQSET_MIN_MAGNITUDE}), ' +
        f'(proportion > {FREQSET_MIN_PROPORTION * 100}%)'
    )
    for freqset in freqsets:
        [syllable, pairs, _] = freqset
        proportion = get_freqset_proportion(freqset)
        sorted_pairs = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)
        pair_strings = [f'{k} = {str(v).ljust(5)}' for [k, v] in sorted_pairs]
        print(
            '{syllable: <16}{prop: <16}{pairs}'.format(
                syllable=f'-{syllable}',
                prop=f'{round(proportion * 100, 2)}% {sorted_pairs[0][0]}',
                pairs=f'{" ".join(pair_strings)}'
            )
        )


def run_analysis(
    genders_and_syllables,
):
    freqsets = defaultdict(lambda: defaultdict(int))
    for [gender, syllable] in genders_and_syllables:
        freqsets[syllable][gender] += 1
    freqsets_items = freqsets.items()
    clean_freqsets = [
        freqset for freqset in freqsets_items
        if is_nice_freqset(freqset)
    ]
    clean_freqsets = clean_freqsets[:N_FREQSETS]
    sorted_clean_freqsets = sorted(
        clean_freqsets,
        key=get_freqset_relevance,
        reverse=True
    )
    freqsets_with_likeliest = [
        [syllable, pairs, max(pairs, key=pairs.get)]
        for [syllable, pairs] in sorted_clean_freqsets
    ]
    return freqsets_with_likeliest


def find_freqset_for_word(freqsets, word):
    # NOTE: Actually comparing with the last syllable gives slightly worse
    # results. This is OK, as people would most likely just check if the word
    # ends with the suffix at any rate.
    # last_syllable = dic.inserted(word).split('-')[-1]
    match = next((
        freqset for freqset in freqsets
        if word.endswith(freqset[0])
        # if last_syllable == freqset[0]
    ), None)
    return match


def match_freqwords(freqsets, freqwords, gendermap):
    def match_freqword(freqword):
        [word, freq] = freqword
        actual_gender = gendermap.get(word)
        freqset = find_freqset_for_word(freqsets, word)
        if not freqset:
            # print(word + ' ' + str(False))
            return False
        [syllable, pairs, likeliest_gender] = freqset
        can_predict = (likeliest_gender == actual_gender)
        # print(word + ' ' + str(can_predict))
        return can_predict
    matches = [
        match_freqword(freqword) for freqword in freqwords
    ]
    accuracy = round(np.sum(np.array(matches) == True) / len(matches), 4)  # noqa
    return [accuracy, matches]


def run():
    [nouninfo, gendermap] = data.load_nouninfo()
    nouninfo = [
        d for d in nouninfo
        if get_gender(d) != 'pl'
    ]
    genders_and_syllables = get_genders_and_last_syllables(nouninfo)

    [gender_to_id, id_to_gender, unique_genders] = get_value_map(
        [d[0] for d in genders_and_syllables]
    )
    [syllable_to_id, id_to_syllable, unique_syllables] = get_value_map(
        [d[1] for d in genders_and_syllables]
    )

    [all_features, all_labels] = make_features_and_labels(
        nouninfo, gender_to_id, syllable_to_id
    )

    #
    # 1. Cross-validation
    #

    print(f'=== Running cross-validation')
    accuracy = run_crossvalidation(
        genders_and_syllables,
        gender_to_id, id_to_gender, unique_genders,
        syllable_to_id, id_to_syllable, unique_syllables,
        all_features, all_labels
    )

    #
    # 2. Frequency analysis
    #

    # print(f'=== Running frequency analysis')
    # freqsets = run_analysis(
    #     genders_and_syllables,
    # )
    # print_freqsets(freqsets)

    #
    # 3. Freqwords analysis
    #

    # print(f'=== Running freqwords analysis')
    # freqwords = data.get_freqwords(N_FREQWORDS)
    # [accuracy, matches] = match_freqwords(freqsets, freqwords, gendermap)
    # print_freqwords_matches(freqwords_matches)
    matches = None
    freqsets = None
    return [accuracy, matches, freqsets]


if __name__ == '__main__':
    # freqset_min_magnitudes = range(0, 2000 + 1, 100)
    # freqset_min_proportions = [x / 10 for x in range(1, 10 + 1, 1)]
    # n_freqsets = range(0, 120 + 1, 10)
    n_folds = range(2000, 200000, 10000)

    freqset_min_magnitudes = [1400]
    freqset_min_proportions = [0.7]
    n_freqsets = [50]

    paramsets = list(itertools.product(
        freqset_min_magnitudes,
        freqset_min_proportions,
        n_freqsets,
        n_folds,
    ))

    print(f'===== Running for {len(paramsets)} paramsets.')

    results = []
    for paramset in paramsets:
        [
            freqset_min_magnitude,
            freqset_min_proportion,
            n_freqsets,
            n_folds,
        ] = paramset

        # so sorry
        FREQSET_MIN_MAGNITUDE = freqset_min_magnitude
        FREQSET_MIN_PROPORTION = freqset_min_proportion
        N_FREQSETS = n_freqsets
        N_FOLDS = n_folds

        [accuracy, matches, freqsets] = run()
        result = [paramset, accuracy]
        results.append(result)
        print(f'>>>>> Finished run: {paramset} {round(accuracy * 100, 2)}%')  # noqa
        # if freqsets:
        #     print_freqsets(freqsets)
        print()

    print()
    print('>>>>> Final results:')
    print(results)
