import time
from collections import defaultdict
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
import data


GRAPHS_DIR = 'graphs/'
FREQSET_MIN_MAGNITUDE = 1000
FREQSET_MIN_PROPORTION = 0.7


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

    n_splits = 5
    print(f'>>> running {n_splits} splits')
    kfold = KFold(n_splits=n_splits)
    kfold.get_n_splits(all_features)

    for idx, [train_index, test_index] in enumerate(kfold.split(all_features)):
        print(f'>> fold {idx}')
        train_features = all_features[train_index]
        test_features = all_features[test_index]
        train_labels = all_labels[train_index]
        test_labels = all_labels[test_index]
        [stats, clf] = run_test(
            train_features, train_labels,
            test_features, test_labels,
        )
        print(f'Accuracy: {round(stats["accuracy"] * 100, 2)}%')
        if idx == 0:
            print('Exporting graph...')
            export_graph(clf, unique_genders)


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
    return (freqset_magnitude / 100000) * freqset_proportion


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
        [syllable, pairs] = freqset
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
    match = next((
        freqset for freqset in freqsets
        if word.endswith(freqset[0])  # Hmm, should we split the word into syllables?
    ), None)
    return match


def eval_freqwords(freqsets, freqwords, gendermap):
    for [word, freq] in freqwords:
        actual_gender = gendermap.get(word)
        freqset = find_freqset_for_word(freqsets, word)
        if freqset:
            [syllable, pairs, likeliest_gender] = freqset
            print(word)
            print(actual_gender)
            print(likeliest_gender)
            print(freqset)
        else:
            print(word)
            print(actual_gender)
        print()


if __name__ == '__main__':
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

    # print(f'=== Running cross-validation')
    # run_crossvalidation(
    #     genders_and_syllables,
    #     gender_to_id, id_to_gender, unique_genders,
    #     syllable_to_id, id_to_syllable, unique_syllables,
    #     all_features, all_labels
    # )

    #
    # 2. Frequency analysis
    #

    print(f'=== Running frequency analysis')
    freqsets = run_analysis(
        genders_and_syllables,
    )
    # print_freqsets(freqsets)

    #
    # 3. Freqwords analysis
    #

    print(f'=== Running freqwords analysis')
    freqwords = data.get_freqwords(10)
    eval_freqwords(freqsets, freqwords, gendermap)
