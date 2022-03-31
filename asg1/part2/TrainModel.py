import sys
import re
from joblib import dump, load
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import re
import utils
import time
import numpy as np
import LightMLETrain

g_dict_vectorize = None
g_tag_key_index = None


def map_features_to_vectors(features_f_name, feature_map_file_name):
    global g_dict_vectorize
    global g_tag_key_index
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
                x_key, x_val = x.split("=", 1)
                # if x_key == "w_not_feature":
                #     keys = [x_val, x_val.lower(), utils.create_word_signature(x_val)]
                #     for key in keys:
                #         tags = known_word_tags.get(key, [])
                #         tags.append(r[0])
                #         known_word_tags[key] = tags
                # else:
                feature_dict[x_key] = x_val
            X.append(feature_dict)
    v = DictVectorizer(sparse=True)
    v = v.fit(X)
    g_dict_vectorize = v
    g_tag_key_index = tag_index_dict
    # save the mapping X into feature_map_file
    # save the known tags dic into feature_map_file
    with open(feature_map_file_name, 'wb') as outfile:
        data_to_dump = {"v": v, "tag_index_dict": tag_index_dict, "known_word_tags": known_word_tags,
                        "mle_estimates": LightMLETrain.serializae_counters()}
        pickle.dump(data_to_dump, outfile)
    X = v.transform(X)

    return X, y


def train_model(X, y):
    classifier = LogisticRegression()
    classifier = classifier.fit(X, y)
    return classifier


def save_model(clf, model_f_name):
    dump(clf, model_f_name)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("invalid args, expected: features_file model_file (optional: feature_map_file, q.mle, e.mle) exiting...")
        exit(1)
    features_f_name = sys.argv[1]
    model_f_name = sys.argv[2]
    feature_map_file_name = "feature_map_file"
    q_mle_file = "q.mle"
    e_mle_file = "e.mle"
    if len(sys.argv) > 3:
        feature_map_file_name = sys.argv[3]
    if len(sys.argv) > 4:
        q_mle_file = sys.argv[4]
    if len(sys.argv) > 5:
        e_mle_file = sys.argv[5]
    total_start = time.perf_counter()
    LightMLETrain.build_transition_counters(q_mle_file)
    LightMLETrain.build_emission_counters(e_mle_file)
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
