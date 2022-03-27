import sys
import MLETrain
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
    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            for word in line.strip().split(" "):
                if k == 200:
                    break
                word_training_tags = MLETrain.get_tags(word)
                if word[-3:] == 'ing':
                    word_training_tags.add("VBG")
                elif any(map(lambda c: c == '-', word)):
                    word_training_tags.add("JJ")
                elif len(word) > 1 and all(map(lambda c: c.isupper(), word)):
                    word_training_tags.add("NNP")
                elif sum(map(lambda c: (1 if c.isdigit() else 0), word)) > float(len(word)) / 2:
                    word_training_tags.add("CD")
                elif len(word) > 0 and word[0].isupper():
                    word_training_tags.add("NNP")
                    word_training_tags.add("NN")
                    word_training_tags.add("NNS")
                elif word[-4:] == 'able':
                    word_training_tags.add("JJ")
                elif word[-2:] == 'ly':
                    word_training_tags.add("RB")
                elif word[-3:] == 'ers':
                    word_training_tags.add("NNS")
                elif word[-4:] == 'tion' or word[-3:] == 'ist' or word[-2:] == 'ty':
                    word_training_tags.add("NN")
                elif len(word_training_tags) == 0:
                    word_training_tags = set(MLETrain.g_tag_dic.keys())
                new_prev_prev_tag_set = set()
                new_prev_tag_set = set()
                for u in prev_prev_tag_set:
                    for v in prev_tag_set:
                        p_max = 0
                        p_argmax = None
                        for w in word_training_tags:
                            p_val = p.get((k - 1, v, u), 1)
                            p_canidate = p_val * MLETrain.get_e((word, w), e_lambda) \
                                         * MLETrain.get_q(w, v, u, lambda1=q_lambda1, lambda2=q_lambda2)
                            if p_canidate > p_max:
                                p_max = p_canidate
                                p_argmax = w
                            new_prev_tag_set.add(w)
                        new_prev_prev_tag_set.add(v)
                        kkey = (k, v, u)
                        if k == 35517 or k == 35516 or k == 35515 or k == 35514:
                            print(kkey, p_argmax, p_max, word)
                    p[kkey] = p_max
                    b[kkey] = (word, p_argmax)
                prev_tag_set = new_prev_tag_set
                prev_prev_tag_set = new_prev_prev_tag_set
                k += 1
    end = time.process_time()
    print(f"time for alg: {end - start} secs\n")
    r_stck = []
    p_argmax = None
    p_max = -1
    prev_result_tag = None
    prev_param_tag = None
    for u in MLETrain.g_tag_dic.keys():
        for v in MLETrain.g_tag_dic.keys():
            key = (k - 1, v, u)
            p_cur = p.get(key, -1)
            if p_cur > p_max:
                p_argmax = b[key]
                prev_result_tag = v
                prev_param_tag = u
                p_max = p_cur
    r_stck.append(p_argmax)
    for i in range(k - 2, -1, -1):
        p_argmax = None
        p_max = -1
        new_prev_param_tag = None
        new_prev_result_tag = None
        for u in MLETrain.g_tag_dic.keys():
            key = (i, prev_param_tag, u)
            p_cur = p.get(key, -1)
            b_cur = b.get(key, None)
            if b_cur is not None:
                result = b_cur[1]
                if result == prev_result_tag:
                    if p_cur > p_max:
                        p_argmax = b_cur
                        p_max = p_cur
                        new_prev_param_tag = prev_result_tag
                        new_prev_result_tag = u
        if new_prev_param_tag is not None:
            prev_param_tag = new_prev_param_tag
            prev_result_tag = new_prev_result_tag
            r_stck.append(p_argmax)
        else:
            # new search critria
            for u in MLETrain.g_tag_dic.keys():
                for v in MLETrain.g_tag_dic.keys():
                    key = (i, v, u)
                    p_cur = p.get(key, -1)
                    if p_cur > p_max:
                        p_argmax = b[key]
                        prev_result_tag = v
                        prev_param_tag = u
                        p_max = p_cur
            r_stck.append(p_argmax)
    print(r_stck)

    # r_stck = []
    # p_arg_max = None
    # p_max = -1
    # prev_tag = None
    # for u in MLETrain.g_tag_dic.keys():
    #     for v in MLETrain.g_tag_dic.keys():
    #         key = (k - 1, v, u)
    #         p_cur = p.get(key, -1)
    #         if p_cur > p_max:
    #             prev_tag = u
    #             p_arg_max = b[key]
    #             p_max = p_cur
    # print(p_arg_max)
    # r_stck.append(p_arg_max)
    # print("###")
    # for i in range(k - 2, 0, -1):
    #     p_max = -1
    #     new_p_arg_max = None
    #     for v in MLETrain.g_tag_dic.keys():
    #         key = (i, v, p_arg_max[1])
    #         p_cur = p.get(key, -1)
    #         b_cur = b.get(key, None)
    #         prev_tag_canidate = None
    #         if p_cur > p_max:
    #             if b_cur is not None:
    #                 print(p_cur, b_cur)
    #                 if b_cur[1] == prev_tag:
    #                     new_p_arg_max = b_cur
    #                     prev_tag_canidate = v
    #                     p_max = p_cur
    #         prev_tag = prev_tag_canidate
    #         p_arg_max = new_p_arg_max
    #     print(prev_tag, p_arg_max, p_max)
    #     r_stck.append(p_arg_max)


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
