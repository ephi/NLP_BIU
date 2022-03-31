import sys
import pickle
import time

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

g_word_set = set()


def load_features(features_f_name):
    with open(features_f_name, "r", encoding="utf-8") as featuresFile:
        lines = featuresFile.readlines()

    words_features = []
    labels = []
    for i, line in enumerate(lines):
        features, label = line.strip().rsplit(' ', 1)
        features = features.split(' ')
        pairs = [tuple(pair.rsplit('=', 1)) for pair in features]
        features_dict = dict(pairs)
        g_word_set.add(features_dict.get('w', ''))
        words_features.append(features_dict)
        labels.append(label)

    return words_features, labels


def to_vectors(features):
    d = DictVectorizer()
    feature_vectors = d.fit_transform(features)
    with open('feature_map_file', 'wb') as file:
        pickle.dump(d, file)
        pickle.dump(g_word_set, file)

    return feature_vectors


def train(features_f_name, model_f_name):
    features, labels = load_features(features_f_name)
    feature_vectors = to_vectors(features)
    logistic_reg = LogisticRegression(random_state=0, max_iter=1000, multi_class='multinomial')
    logistic_reg.fit(feature_vectors, labels)
    pickle.dump(logistic_reg, open(model_f_name, 'wb'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("invalid args, expected: features_f_name model_f_name exiting...")
        exit(1)
    features_f_name = sys.argv[1]
    model_f_name = sys.argv[2]
    start = time.perf_counter()
    train(features_f_name, model_f_name)
    print("Elapsed time: %.2f [m]" % ((time.perf_counter() - start) / 60))