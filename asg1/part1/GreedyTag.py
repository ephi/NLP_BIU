import sys
import MLETrain
from MLETrain import ExtraFileOpts
import AccuracyChecker
import numpy as np


def update_window(window, word):
    window[1] = window[0]
    window[0] = word


def report_accuracy(opts):
    if opts is None:
        print("options must be used to report accuracy")
        return
    expctd_f_name = opts.get_option("dev_test")
    if expctd_f_name is None:
        print("dev_test must be configured in options in order to report accuracy")
        return
    pred_f_name = opts.get_option("pred_file")
    if expctd_f_name is None:
        print("pred_file must be configured in options in order to report accuracy")
        return
    acc = AccuracyChecker.accuracy_on_file(pred_f_name, expctd_f_name)
    print(f"{pred_f_name} vs. {expctd_f_name}, accuracy: {acc}")


def predict_tags(input_f_name, output_f_name, opts=None):
    e_lambda = None
    q_lambda1 = None
    q_lambda2 = None
    q_lambda3 = None
    if opts is not None:
        e_lambda = opts.get_option("e_lambda")
        q_lambda1 = opts.get_option("q_lambda1")
        q_lambda2 = opts.get_option("q_lambda2")
        q_lambda3 = opts.get_option("q_lambda3")

    with open(input_f_name, 'r', encoding='utf-8') as file:
        with open(output_f_name, 'w', encoding='utf-8') as outfile:
            for line in file.readlines():
                line = line.strip()
                window = [MLETrain.START_TAG, MLETrain.START_TAG]
                str_r = ""
                for i, word in enumerate(line.strip().split(" ")):
                    max_tag_p = float('-inf')
                    max_tag_v = ""
                    if word.lower() not in MLETrain.g_tag_per_word_dic.keys():
                        word = MLETrain.create_word_signature(word)
                        if word == '^UNK':
                            str_r += f"{word}/{'*UNK*'} "
                            update_window(window, '*UNK*')
                            continue
                    else:
                        word = word.lower()
                    for tag in MLETrain.g_tag_per_word_dic[word]:
                        if tag == MLETrain.START_TAG or tag == MLETrain.END_TAG:
                            continue
                        cur_tag_p = (np.log(MLETrain.get_e((word, tag))) +
                                    np.log(MLETrain.get_q(tag, window[0], window[1], q_lambda1, q_lambda2, q_lambda3)))
                        if cur_tag_p > max_tag_p:
                            max_tag_p = cur_tag_p
                            max_tag_v = tag
                    str_r += f"{word}/{max_tag_v} "
                    update_window(window, max_tag_v)
                str_r = str_r.rstrip() + "\n"
                outfile.write(str_r)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("invalid args, expected: input_file_name q.mle e.mle greedy_hmm_output.txt extra_file.txt exiting...")
        exit(1)
    input_f_name = sys.argv[1]
    q_mle_f_name = sys.argv[2]
    e_mle_f_name = sys.argv[3]
    output_f_name = sys.argv[4]
    extra_f_name = sys.argv[5]
    MLETrain.build_transition_counters(q_mle_f_name)
    MLETrain.build_emission_counters(e_mle_f_name)
    opts = ExtraFileOpts(extra_f_name)
    opts.set_option("pred_file", output_f_name)
    predict_tags(input_f_name, output_f_name, opts)
    report_accuracy(opts)
