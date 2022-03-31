import sys
import re
import random

g_word_freq_dic = {}
g_word_tag_pair_list = []


def build_word_freq_dic_from_input(input_f_name):
    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            for s in line.strip().split(" "):
                g_word_freq_dic[s] = g_word_freq_dic.get(s, 0) + 1


def build_word_dicts_from_input(input_f_name: str):
    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            start_pair = ("startline", "<S>")
            g_word_tag_pair_list.append(start_pair)
            g_word_tag_pair_list.append(start_pair)
            for s in line.strip().split(" "):
                r = s.rsplit('/', 1)
                word = r[0]
                g_word_freq_dic[word] = g_word_freq_dic.get(word, 0) + 1
                pair = (word, r[1])
                g_word_tag_pair_list.append(pair)
            end_pair = ("endline", "<E>")
            g_word_tag_pair_list.append(end_pair)
            g_word_tag_pair_list.append(end_pair)


def is_rare(word):
    return g_word_freq_dic.get(word, 0) <= 4


def add_suffixes_to_dict(word, fix_name, fix_len, dict):
    for i in range(1, fix_len + 1):
        dict["s" + fix_name + str(i)] = word[-i:]


def add_postfixes_to_dict(word, fix_name, fix_len, dict):
    for i in range(1, fix_len + 1):
        dict["p" + fix_name + str(i)] = word[:i]


def add_specials_to_dict(word, name, dict):
    # if entry is not found at dic, then, feature doesn't exist

    dict['c_num'] = str(any(char.isdigit() for char in word))
    dict['c_hyphen'] = str(any(char == '-' for char in word))
    dict['c_upper'] = str(any(char.isupper() for char in word))


def extract_per_pair(pair, name, dict, allow_tag=True, allow_fix=True):
    word = pair[0]
    tag = pair[1]
    if allow_tag:
        dict[name + "t"] = tag
    if not is_rare(word) or not allow_fix:
        dict[name + "w"] = word.lower()
    elif allow_fix:
        lower_word = word.lower()
        fix_len = min(len(word), 3)
        add_suffixes_to_dict(lower_word, name, fix_len, dict)
        add_postfixes_to_dict(lower_word, name, fix_len, dict)
        add_specials_to_dict(word, name, dict)


def extract(i, word_tag_pair_list):
    dict = {}
    cur_pair = word_tag_pair_list[i]
    prev_prev_pair = word_tag_pair_list[i - 2]
    prev_pair = word_tag_pair_list[i - 1]
    future_pair = word_tag_pair_list[i + 1]
    future_future_pair = word_tag_pair_list[i + 2]

    word = cur_pair[0].lower()
    if not is_rare(word):
        dict["w"] = word
    else:
        add_suffixes_to_dict(word, "", 4, dict)
        add_postfixes_to_dict(word, "", 4, dict)
        add_specials_to_dict(word, "", dict)
    dict["pw"] = prev_pair[0].lower()
    dict["ppw"] = prev_prev_pair[0].lower()
    dict["fw"] = future_pair[0].lower()
    dict["ffw"] = future_future_pair[0].lower()
    dict["ppt_pt"] = prev_prev_pair[1] + "|" + prev_pair[1]

    # extract_per_pair(prev_prev_pair, "pp", dict, allow_tag=True)
    # extract_per_pair(prev_pair, "p", dict, allow_tag=True)
    # extract_per_pair(cur_pair, "c", dict, allow_tag=False)
    # # dict["ppt_pt"] = prev_prev_pair[1] + "|" + prev_pair[1]
    # extract_per_pair(future_pair, "f", dict, False)
    # extract_per_pair(future_future_pair, "ff", dict, False)
    # # dict["w_not_feature"] = cur_pair[0]
    # # if not is_rare(prev_pair[0]) and not is_rare(prev_prev_pair[0]):
    # # dict["ppt_pt_ct_ft_fft"] = prev_prev_pair[1] + "|" + prev_pair[1] + "|" \
    # #                           + cur_pair[1] + "|" + future_pair[1] + "|" + future_future_pair[1]
    # # if not is_rare(future_pair[0]) and not is_rare(future_future_pair[0]):
    # # dict["ft_fft"] = future_pair[1] + "|" + future_future_pair[1]
    return dict


def write_features_to_output(output_f_name):
    with open(output_f_name, "w") as out_f:
        for i, word_tag_pair in enumerate(g_word_tag_pair_list):
            if word_tag_pair[0] == "startline" or word_tag_pair[0] == "endline":
                continue
            out_feature_line = word_tag_pair[1]
            features = extract(i, g_word_tag_pair_list)
            out = ''.join([key + "=" + val + " " for key, val in features.items()])
            out = out_feature_line + " " + out.strip() + "\n"
            out_f.write(out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("invalid args, expected: input_file_name features_output_file_name exiting...")
        exit(1)
    input_f_name = sys.argv[1]
    output_f_name = sys.argv[2]
    build_word_dicts_from_input(input_f_name)
    write_features_to_output(output_f_name)
