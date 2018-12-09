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
    return value_map


def make_features_and_labels(nouninfo, gender_map, syllable_map):
    """
    Returns [features, labels], where features are:

        [[syllable_id], [...], ...]

    and labels are:

        [gender_id, ...]
    """
    features = [
        [syllable_map[get_last_syllable(d)]]
        for d in nouninfo
    ]
    labels = [
        gender_map[get_gender(d)]
        for d in nouninfo
    ]
    return [features, labels]


if __name__ == '__main__':
    nouninfo = data.load_nouninfo()
    genders_and_syllables = get_genders_and_last_syllables(nouninfo)

    gender_map = get_value_map([d[0] for d in genders_and_syllables])
    gender_map_inv = {v: k for k, v in gender_map.items()}

    syllable_map = get_value_map([d[1] for d in genders_and_syllables])
    syllable_map_inv = {v: k for k, v in syllable_map.items()}

    [features, labels] = make_features_and_labels(
        nouninfo, gender_map, syllable_map
    )

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    pred = clf.predict([[syllable_map['chen']]])
    print(pred)
    print(gender_map_inv[pred[0]])
