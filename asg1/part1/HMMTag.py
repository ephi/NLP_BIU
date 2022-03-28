import sys
import MLETrain
import numpy as np
from MLETrain import ExtraFileOpts
import AccuracyChecker
import time


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


def safe_log(a):
    if a <= 0:
        return float("-inf")
    return np.log(a)


def predict_tags(input_f_name, output_f_name, opts=None):
    e_lambda = None
    q_lambda1 = None
    q_lambda2 = None
    if opts is not None:
        e_lambda = opts.get_option("e_lambda")
        q_lambda1 = opts.get_option("q_lambda1")
        q_lambda2 = opts.get_option("q_lambda2")
    # Viterbi init
    b = {}
    p = {}

    # Viterbi pass

    start = time.process_time()
    k = 1
    prev_tag_set = set()
    prev_tag_set.add(None)
    prev_prev_tag_set = set()
    prev_prev_tag_set.add(None)
    p[(0, None, None)] = 0
    words_in_input = []
    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            words = line.strip().split(" ")
            if k == 1:
                words.insert(0, "*START*")
                words.insert(0, "*START*")
            for word in words:
                words_in_input.append(word)
                tag_search_set = MLETrain.get_tags(word)
                if word[-3:] == 'ing':
                    tag_search_set.add("VBG")
                elif len(word) == 1 and (word[0] == "." or word[0] == "?"):
                    tag_search_set.add(".")
                elif len(word) == 1 and word[0] == "$":
                    tag_search_set.add("$")
                elif len(word) == 1 and word[0] == ",":
                    tag_search_set.add(",")
                elif any(map(lambda c: c == '-', word)):
                    tag_search_set.add("JJ")
                elif len(word) > 1 and all(map(lambda c: c.isupper(), word)):
                    tag_search_set.add("NNP")
                elif sum(map(lambda c: (1 if c.isdigit() else 0), word)) > float(len(word)) / 2:
                    tag_search_set.add("CD")
                elif len(word) > 0 and word[0].isupper():
                    tag_search_set.add("NNP")
                    tag_search_set.add("NN")
                    tag_search_set.add("NNS")
                elif word[-4:] == 'able':
                    tag_search_set.add("JJ")
                elif word[-2:] == 'ly':
                    tag_search_set.add("RB")
                elif word[-3:] == 'ers':
                    tag_search_set.add("NNS")
                elif word[-4:] == 'tion' or word[-3:] == 'ist' or word[-2:] == 'ty':
                    tag_search_set.add("NN")
                elif len(tag_search_set) == 0:
                    tag_search_set = set(MLETrain.g_tag_dic.keys())
                for u in prev_tag_set:
                    for v in tag_search_set:
                        p_max = float("-inf")
                        p_argmax = None
                        for w in prev_prev_tag_set:
                            p_val = p.get((k - 1, w, u), float("-inf"))
                            p_canidate = p_val + safe_log(MLETrain.get_e((word, v), e_lambda)) + \
                                         safe_log(MLETrain.get_q(v, w, u, lambda1=q_lambda1, lambda2=q_lambda2))
                            if p_canidate > p_max:
                                p_max = p_canidate
                                p_argmax = w
                        kkey = (k, u, v)

                        p[kkey] = p_max
                        b[kkey] = p_argmax
                prev_prev_tag_set = prev_tag_set
                prev_tag_set = tag_search_set
                k += 1
            words_in_input.append("\n")

    end = time.process_time()
    print(f"time for alg: {end - start} secs\n")

    m_key = None
    n = k - 1
    p_max = float("-inf")
    for u in MLETrain.g_tag_dic.keys():
        for v in MLETrain.g_tag_dic.keys():
            canidate_key = (n, u, v)
            p_val = p.get(canidate_key, float("-inf"))
            # p_canidate = p_val + \
            #             safe_log(MLETrain.get_q(v, u, None, lambda1=q_lambda1, lambda2=q_lambda2))
            if p_val > p_max:
                p_max = p_canidate
                m_key = canidate_key
    print(m_key, p[m_key], b[m_key])
    r_stck = [m_key[2], m_key[1]]
    for i, k in enumerate(range(n - 2, 0, -1)):
        r_stck.append(b[(k + 2, r_stck[i + 1], r_stck[i])])
    r_stck.reverse()

    with open(output_f_name, 'w', encoding='utf-8') as file:
        str_r = ""
        k = 2
        for word in words_in_input:
            if word == "*START*":
                continue
            if word == "\n":
                str_r = str_r.rstrip()
                str_r += "\n"
            else:
                str_r += word + "/" + r_stck[k] + " "
                k += 1
        str_r = str_r.rstrip()
        file.write(str_r)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("invalid args, expected: input_file_name q.mle e.mle viterbi_hmm_output.txt extra_file.txt exiting...")
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
