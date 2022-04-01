import sys
import re
import pickle
import numpy as np
from joblib import dump, load
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from ExtractFeatures import extract
from collections import Counter
import time


def load_model(mdl_f_name):
    clf = load(mdl_f_name)
    return clf


def get_most_freq_tag(tags):
    occurence_count = Counter(tags)
    return occurence_count.most_common(1)[0][0]


def load_model_input(input_f_name, feat_map_f_name):
    # load additional information from feature map file
    with open(feat_map_f_name, 'rb') as file:
        data_dic = pickle.load(file)
        v = data_dic["v"]
        # known_word_tags = data_dic["known_word_tags"]
        tag_index_dict = data_dic["tag_index_dict"]

    word_tag_pair_list = []
    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            start_pair = ("startline", "<S>")
            word_tag_pair_list.append(start_pair)
            word_tag_pair_list.append(start_pair)
            for s in line.strip().split(" "):
                r = s.rsplit("/", 1)[0]
                pair = (r, None)
                word_tag_pair_list.append(pair)
            end_pair = ("endline", "<E>")
            word_tag_pair_list.append(end_pair)
            word_tag_pair_list.append(end_pair)
    X = []
    for i, word_tag_pair in enumerate(word_tag_pair_list):
        if word_tag_pair[0] == "startline" or word_tag_pair[0] == "endline":
            continue
        features = extract(i, word_tag_pair_list)
        # del features["w_not_feature"]
        X.append(features)
    X = v.transform(X)
    return X, word_tag_pair_list, tag_index_dict


def generative_predictor(clf, input_f_name, feat_map_f_name):
    # load additional information from feature map file
    with open(feat_map_f_name, 'rb') as file:
        data_dic = pickle.load(file)
        v = data_dic["v"]
        tag_index_dict = data_dic["tag_index_dict"]

    # inverse the tag index dict
    tag_index_dict = {v: k for k, v in tag_index_dict.items()}

    # build word list *without* tags
    word_tag_pair_list = []
    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            start_pair = ["startline", "<S>"]
            word_tag_pair_list.append(start_pair)
            word_tag_pair_list.append(start_pair)
            for s in line.strip().split(" "):
                pair = [s, None]
                word_tag_pair_list.append(pair)
            end_pair = ["endline", "<E>"]
            word_tag_pair_list.append(end_pair)
            word_tag_pair_list.append(end_pair)

    # start the generative prediction
    allsize = len(word_tag_pair_list)
    for i, word_tag_pair in enumerate(word_tag_pair_list):
        if word_tag_pair[0] == "startline" or word_tag_pair[0] == "endline":
            continue
        features = extract(i, word_tag_pair_list)
        # del features["w_not_feature"]
        x = v.transform(features)
        y_hat = clf.predict_log_proba(x)
        y_hat = np.argmax(y_hat)
        y_hat = tag_index_dict[y_hat]
        word_tag_pair_list[i][1] = y_hat
        if i % 1000 == 0:
            print(f"{(i / allsize) * 100}% of predication completed.")

    return word_tag_pair_list


def write_prediction_output_from_pairs(pred_f_name, word_list_pairs):
    with open(pred_f_name, 'w', encoding='utf-8') as file:
        str_res = ""
        for i, pair in enumerate(word_list_pairs):
            if pair[0] == "endline":
                continue
            if pair[0] == "startline":
                if i > 0 and word_list_pairs[i - 1][0] == "endline":
                    str_res = str_res.rstrip()
                    str_res += "\n"
                    file.write(str_res)
                    str_res = ""
                continue
            pred = pair[1]
            str_res += pair[0] + "/" + pred + " "
        str_res = str_res.rstrip()
        file.write(str_res)


def write_prediction_output(pred_f_name, word_list, y_hat):
    with open(pred_f_name, 'w', encoding='utf-8') as file:
        str_res = ""
        y_hat_runner = 0
        for i, pair in enumerate(word_list):
            if pair[0] == "endline":
                continue
            if pair[0] == "startline":
                if i > 0 and word_list[i - 1][0] == "endline":
                    str_res = str_res.rstrip()
                    str_res += "\n"
                    file.write(str_res)
                    str_res = ""
                continue
            pred = y_hat[y_hat_runner]
            str_res += pair[0] + "/" + pred + " "
            y_hat_runner += 1
        str_res = str_res.rstrip()
        file.write(str_res)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("invalid args, expected: input_file_name model_file_name feature_map_file output_file exiting...")
        exit(1)
    input_f_name = sys.argv[1]
    mdl_f_name = sys.argv[2]
    feat_map_f_name = sys.argv[3]
    pred_f_name = sys.argv[4]
    total_start = time.perf_counter()
    # load the model
    start = time.perf_counter()
    X, word_list, tag_index_dict = load_model_input(input_f_name, feat_map_f_name)
    end = time.perf_counter()
    print(f"load_model_input took {(end - start)} secs")
    # load the classifier
    classifier = load_model(mdl_f_name)
    start = time.perf_counter()
    # predict y_hat & prepared it as tag list
    y_hat = classifier.predict_log_proba(X)
    y_hat = np.argmax(y_hat, axis=1)
    y_hat_as_tags = []
    # invert the tag_index_dict
    tag_index_dict = {v: k for k, v in tag_index_dict.items()}
    for p in y_hat:
        y_hat_as_tags.append(tag_index_dict[p])
    end = time.perf_counter()
    print(f"prediction took {(end - start)} secs")
    start = time.perf_counter()
    write_prediction_output(pred_f_name, word_list, y_hat_as_tags)
    end = time.perf_counter()
    print(f"write_prediction_output took {(end - start)} secs")
    total_end = time.perf_counter()
    print(f"entire program execution took {(total_end - total_start)} secs")

    # build counts to extract features properly.
    # classifier = load_model(mdl_f_name)
    # start = time.perf_counter()
    # word_list_pairs = generative_predictor(classifier, input_f_name, feat_map_f_name)
    # end = time.perf_counter()
    # print('Elapsed time: ', (end - start))
    # write_prediction_output_from_pairs(pred_f_name, word_list_pairs)
