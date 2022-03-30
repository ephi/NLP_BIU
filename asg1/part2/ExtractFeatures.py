import sys
import re

g_word_freq_dic = {}
g_word_tag_pair_list = []


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
    return g_word_freq_dic.get(word, 0) <= 3


def add_suffixes_to_dict(word, fix_len, dict):
    for i in range(1, fix_len + 1):
        dict["s" + str(i)] = word[-i:]


def add_postfixes_to_dict(word, fix_len, dict):
    for i in range(1, fix_len + 1):
        dict["p" + str(i)] = word[:i]


def add_specials_to_dict(word, dict):
    # if entry is not found at dic, then, feature doesn't exist

    # digits
    if re.search(r'^[0-9]+[,/.][0-9]+[,]?[0-9]*$', word) is not None:
        dict["c_digit"] = "T"
    # time
    if re.search(r'^[0-9]+:[0-9]+$', word) is not None:
        dict["c_time"] = "T"
    # fruc
    if re.search(r'^[0-9]+/[0-9]+-[a-zA-Z]+[-]?[a-zA-Z]*$', word) is not None:
        dict["c_fruc"] = "T"
    # upper case
    if re.search(r'^[A-Z]+$', word) is not None:
        dict["c_upper"] = "T"


# prev_words_pairs[0] = (w_i-2, tag(w_i-2))
# prev_words_pairs[1] = (w_i-1, tag(w_i-1))
# word = w_i
# future_words_pairs[0] =(w_i + 1, tag(w_i + 1))
# future_words_pairs[1] = (w_i + 2, tag(w_i + 2))
def extract(word, i, word_tag_pair_list):
    dict = {}
    prev_prev_pair = word_tag_pair_list[i - 2]
    prev_pair = word_tag_pair_list[i - 1]
    future_pair = word_tag_pair_list[i + 1]
    future_future_pair = word_tag_pair_list[i + 2]
    dict["pw"] = prev_pair[0]
    dict["ppw"] = prev_prev_pair[0]
    dict["pt"] = prev_pair[1]
    dict["ppt_pt"] = prev_prev_pair[1] + " " + dict["pt"]
    dict["fw"] = future_pair[0]
    dict["ffw"] = future_future_pair[0]
    if not is_rare(word):
        dict["w"] = word
    else:
        add_suffixes_to_dict(word, 3, dict)
        add_postfixes_to_dict(word, 3, dict)
        add_specials_to_dict(word, dict)
    return dict


def write_features_to_output(output_f_name):
    with open(output_f_name, "w") as out_f:
        for i, word_tag_pair in enumerate(g_word_tag_pair_list):
            if word_tag_pair[0] == "startline" or word_tag_pair[0] == "endline":
                continue
            out_feature_line = word_tag_pair[1]
            features = extract(word_tag_pair[0], i, g_word_tag_pair_list)
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
