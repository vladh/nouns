import pickle
import pyphen
import config as config


NOUNS_PATH = f'{config.DATA_DIR}leo/leo-nouns-uniq.txt'
CACHE_DIR = 'cache/'
CACHED_NOUNINFO_PATH = f'{CACHE_DIR}nouninfo.pkl'


def parse_noun_line(line):
    """
    Returns [word, gender] for a Leo dictionary noun entry line.
    Format: (aktive) Langzeitverbindung {f}
    """
    gender_start = line.find('{') + 1
    gender_end = line.find('}')
    word_end = line.find('{') - 1
    paren_end = line.find(')')
    word_start = paren_end + 2 if paren_end > -1 else 0
    word = line[word_start:word_end]
    gender = line[gender_start:gender_end]
    return [word, gender]


def get_nouninfo():
    """
    Returns a list of the form:

    [['Regel', 'f', 're-gel'], ...]
    """
    dic = pyphen.Pyphen(lang='de_DE')
    nouninfo = []
    with open(NOUNS_PATH, 'r') as lines:
        for line in lines:
            [word, gender] = parse_noun_line(line)
            separation = dic.inserted(word)
            nouninfo.append([word, gender, separation])
    return nouninfo


def save_nouninfo():
    nouninfo = get_nouninfo()
    file = open(CACHED_NOUNINFO_PATH, 'wb')
    pickle.dump(nouninfo, file)


def load_nouninfo():
    file = open(CACHED_NOUNINFO_PATH, 'rb')
    nouninfo = pickle.load(file)
    return nouninfo


if __name__ == '__main__':
    # save_nouninfo()
    nouninfo = load_nouninfo()
    print(nouninfo[5000])
