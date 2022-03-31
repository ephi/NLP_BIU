import sys
import re
from joblib import dump, load
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import re
import time
import numpy as np


def map_features_to_vectors(features_f_name, feature_map_file_name):
    tag_index_dict = {}
    y = []
    X = []
    tag_index = 0
    with open(features_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            r = line.strip().split(" ")
            # skip blank lines
            if len(r) == 0:
                continue
            # map feature label to unique key
            found_tag_index = tag_index_dict.get(r[0], -1)
            nxt_tag_index = tag_index
            if found_tag_index == -1:
                found_tag_index = tag_index
                tag_index_dict[r[0]] = tag_index
                nxt_tag_index += 1
            y.append(found_tag_index)
            tag_index = nxt_tag_index
            # add dict reprsenting the feature
            feature_dict = {}
            discreate_feature = r[1:]
            for x in discreate_feature:
                x_key, x_val = x.split("=", 1)
                feature_dict[x_key] = x_val
            X.append(feature_dict)
    v = DictVectorizer(sparse=True)
    v = v.fit(X)
    # save the mapping X into feature_map_file
    # save the known tags dic into feature_map_file
    with open(feature_map_file_name, 'wb') as outfile:
        data_to_dump = {"v": v, "tag_index_dict": tag_index_dict}
        pickle.dump(data_to_dump, outfile)
    X = v.transform(X)

    return X, y


def train_model(X, y):
    classifier = SGDClassifier(loss='log')
    classifier = classifier.fit(X, y)
    return classifier


def save_model(clf, model_f_name):
    dump(clf, model_f_name)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("invalid args, expected: features_file model_file (optional: feature_map_file) exiting...")
        exit(1)
    features_f_name = sys.argv[1]
    model_f_name = sys.argv[2]
    feature_map_file_name = "feature_map_file"
    if len(sys.argv) > 3:
        feature_map_file_name = sys.argv[3]
    total_start = time.perf_counter()
    start = time.perf_counter()
    X, y = map_features_to_vectors(features_f_name, feature_map_file_name)
    end = time.perf_counter()
    print(f"map_features_to_vectors took {(end - start)} secs")
    start = time.perf_counter()
    clf = train_model(X, y)
    end = time.perf_counter()
    print(f"train_model took {(end - start)} secs")
    save_model(clf, model_f_name)
    total_end = time.perf_counter()
    print(f"entire program execution took {(total_end - total_start)} secs")
