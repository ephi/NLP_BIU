import sys
import re
from joblib import dump, load
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
import re
import utils
import numpy as np


def map_features_to_vectors(features_f_name, feature_map_file_name):
    tag_index_dict = {}
    known_word_tags = {}
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
                x_key, x_val = x.split("~=")
                if x_key == "w_not_feature":
                    keys = [x_val, x_val.lower(), utils.create_word_signature(x_val)]
                    for key in keys:
                        tags = known_word_tags.get(key, [])
                        tags.append(r[0])
                        known_word_tags[key] = tags
                else:
                    try:
                        feature_dict[x_key] = int(x_val)
                    except:
                        feature_dict[x_key] = x_val
            X.append(feature_dict)
    v = DictVectorizer(sparse=True)
    v = v.fit(X)
    # save the mapping X into feature_map_file
    # save the known tags dic into feature_map_file
    with open(feature_map_file_name, 'wb') as outfile:
        data_to_dump = {"v": v, "tag_index_dict": tag_index_dict, "known_word_tags": known_word_tags}
        pickle.dump(data_to_dump, outfile)
    X = v.transform(X)

    return X, y


def train_model(X, y):
    classifier = SGDClassifier(loss="log")
    y = np.array(y)
    for i in range(0, 10):
        idx = np.random.randint(X.shape[0], size=int(X.shape[0] * 0.2))
        Y_batch = y[idx]
        X_batch = X[idx, :]
        cls_cnt = None
        if i == 0:
            cls_cnt = np.unique(y)
        classifier = classifier.partial_fit(X_batch, Y_batch, classes=cls_cnt)
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
    X, y = map_features_to_vectors(features_f_name, feature_map_file_name)
    clf = train_model(X, y)
    save_model(clf, model_f_name)
