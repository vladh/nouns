import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
import data


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
    return stats


def run_crossvalidation():
    nouninfo = data.load_nouninfo()
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
        stats = run_test(
            train_features, train_labels,
            test_features, test_labels,
        )
        print(f'Accuracy: {round(stats["accuracy"] * 100, 2)}%')


if __name__ == '__main__':
    run_crossvalidation()
