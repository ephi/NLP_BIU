import sys
import pickle
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
    return g_word_freq_dic.get(word, 0) <= 10


def add_suffixes_to_dict(word, fix_name, fix_len, dict):
    for i in range(1, fix_len + 1):
        dict["s" + fix_name + str(i)] = word[-i:]


def add_postfixes_to_dict(word, fix_name, fix_len, dict):
    for i in range(1, fix_len + 1):
        dict["p" + fix_name + str(i)] = word[:i]


def add_specials_to_dict(word, name, dict):
    # if entry is not found at dic, then, feature doesn't exist

    dict[name + '_num'] = str(any(char.isdigit() for char in word))
    dict[name + '_hyphen'] = str(any(char == '-' for char in word))
    dict[name + "_upper"] = str(any(char.isupper() for char in word))
    dict[name + '_backslash'] = str(any(char == '/' for char in word))
    dict[name + '_mod'] = str(any(char == '%' for char in word))
    dict[name + '_dollar'] = str(any(char == '$' for char in word))
    dict[name + '_dot'] = str(any(char == '.' for char in word))
    dict[name + "_aupper"] = str(all(char.isupper() for char in word))


def extract(i, word_tag_pair_list):
    dict = {}
    cur_pair = word_tag_pair_list[i]
    prev_prev_pair = word_tag_pair_list[i - 2]
    prev_pair = word_tag_pair_list[i - 1]
    future_pair = word_tag_pair_list[i + 1]
    future_future_pair = word_tag_pair_list[i + 2]

    word = cur_pair[0].lower()
    # 94.9 ######
    # fix_len = min(len(word), 8)
    # c_name = "c"
    # add_suffixes_to_dict(word, c_name, fix_len, dict)
    # add_postfixes_to_dict(word, c_name, fix_len, dict)
    # add_specials_to_dict(cur_pair[0], c_name, dict)
    # dict["pw"] = prev_pair[0]
    # dict["ppw"] = prev_prev_pair[0]
    # dict["fw"] = future_pair[0]
    # dict["ffw"] = future_future_pair[0]
    # dict["pt"] = prev_pair[1]
    # dict["ft"] = future_pair[1]
    # dict["ppt_pt"] = prev_prev_pair[1] + "|" + prev_pair[1]
    # dict["ft_fft"] = future_pair[1] + "|" + future_future_pair[1]
    #################93.9##################
    # fix_len = min(len(word), 9)
    # c_name = "c"
    # add_suffixes_to_dict(word, c_name, fix_len, dict)
    # add_postfixes_to_dict(word, c_name, fix_len, dict)
    # add_specials_to_dict(cur_pair[0], c_name, dict)
    # dict["pw_pt"] = prev_pair[0] + "/" + prev_pair[1]
    # dict["ppw_ppt"] = prev_prev_pair[0] + "/" + prev_prev_pair[1]
    # dict["fw_ft"] = future_pair[0] + "/" + future_pair[1]
    # dict["ffw_fft"] = future_future_pair[0] + "/" + future_future_pair[1]
    #################94.4######################
    fix_len = min(len(word), 12)
    c_name = "c"
    add_suffixes_to_dict(word, c_name, fix_len, dict)
    add_postfixes_to_dict(word, c_name, fix_len, dict)
    add_specials_to_dict(cur_pair[0], c_name, dict)
    dict["w_len"] = str(len(word))
    dict["pw"] = prev_pair[0]
    dict["ppw"] = prev_prev_pair[0]
    dict["fw"] = future_pair[0]
    dict["ffw"] = future_future_pair[0]
    # dict["pt"] = prev_pair[1]
    # dict["ft"] = future_pair[1]
    # dict["ppt_pt"] = prev_prev_pair[1] + "|" + prev_pair[1]
    # dict["ft_fft"] = future_pair[1] + "|" + future_future_pair[1]
    ############################################################

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
