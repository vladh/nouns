import numpy as np
from sklearn import tree
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
    unique_values = np.unique(values)
    value_map = dict(zip(unique_values, range(0, len(unique_values))))
    value_map_inv = dict(zip(range(0, len(unique_values)), unique_values))

    def do_map(d):
        return value_map[d]

    def do_map_inv(d):
        return value_map_inv[d]

    return [do_map, do_map_inv]


def make_features_and_labels(nouninfo, gender_to_id, syllable_to_id):
    """
    Returns [features, labels], where features are:

        [[syllable_id], [...], ...]

    and labels are:

        [gender_id, ...]
    """
    features = [
        [syllable_to_id(get_last_syllable(d))]
        for d in nouninfo
    ]
    labels = [
        gender_to_id(get_gender(d))
        for d in nouninfo
    ]
    return [features, labels]


if __name__ == '__main__':
    nouninfo = data.load_nouninfo()
    genders_and_syllables = get_genders_and_last_syllables(nouninfo)

    [gender_to_id, id_to_gender] = get_value_map(
        [d[0] for d in genders_and_syllables]
    )
    [syllable_to_id, id_to_syllable] = get_value_map(
        [d[1] for d in genders_and_syllables]
    )

    [features, labels] = make_features_and_labels(
        nouninfo, gender_to_id, syllable_to_id
    )

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    pred = clf.predict([[syllable_to_id('chen')]])
    print(pred)
    print(id_to_gender(pred[0]))
