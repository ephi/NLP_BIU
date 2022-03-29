import sys
import MLETrain
from MLETrain import ExtraFileOpts
from GreedyTag import report_accuracy
import time
import numpy as np


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
            tags = list(MLETrain.g_tag_dic.keys())
            num_tags = len(tags)

            for line in file.readlines():
                line = line.strip()
                words = line.strip().split(" ")
                num_words = len(words)
                V = np.ones((num_words, num_tags, num_tags), dtype=float) * float('-inf')
                B = np.empty((num_words, num_tags, num_tags), dtype=object)

                prev_word = 'startline'
                prev_prev_word = 'startline'

                for i, word in enumerate(words):
                    if word not in MLETrain.g_tag_per_word_dic.keys():
                        word = word.lower()
                        if word not in MLETrain.g_tag_per_word_dic.keys():
                            word = MLETrain.create_word_signature(word)
                    # else:
                    #     word = word.lower()
                    possible_tags = MLETrain.g_tag_per_word_dic[word]
                    prev_tag_set = MLETrain.g_tag_per_word_dic[prev_word]
                    prev_prev_tag_set = MLETrain.g_tag_per_word_dic[prev_prev_word]

                    for tag in possible_tags:
                        j = tags.index(tag)
                        prev_tags = [(prev_prev_tag, prev_tag)
                                     for prev_prev_tag in prev_prev_tag_set for prev_tag in prev_tag_set]

                        for (prev_prev_tag, prev_tag) in prev_tags:
                            k = tags.index(prev_tag)
                            l = tags.index(prev_prev_tag)
                            cur_tag_p = ((0 if i == 0 else V[i - 1, k, l]) +
                                         np.log(MLETrain.get_e((word, tag), e_lambda)) +
                                         np.log(MLETrain.get_q(tag, prev_tag, prev_prev_tag, q_lambda1, q_lambda2,
                                                               q_lambda3)))

                            if cur_tag_p > V[i, j, k]:
                                V[i, j, k] = cur_tag_p
                                B[i, j, k] = prev_prev_tag
                        if np.all(V[i, :, :] == float('-inf')):
                            print('k')

                    prev_prev_word = prev_word
                    prev_word = word

                max_tag_p = float('-inf')
                l_max, k_max = None, None
                for k, prev_tag in enumerate(tags):
                    for l, prev_prev_tag in enumerate(tags):
                        cur_tag_p = (V[-1, k, l] +
                                     np.log(MLETrain.get_e(('endline', MLETrain.END_TAG))) +
                                     np.log(MLETrain.get_q(MLETrain.END_TAG, prev_tag, prev_prev_tag, q_lambda1,
                                                           q_lambda2, q_lambda3)))

                        if cur_tag_p > max_tag_p:
                            max_tag_p = cur_tag_p
                            l_max, k_max = l, k

                predicted_tags = []
                for i in range(num_words - 1, -1, -1):
                    predicted_tags.append(tags[k_max])
                    tmp = l_max
                    l_max = tags.index(B[i, k_max, l_max])
                    k_max = tmp
                predicted_tags.reverse()

                str_r = ''
                for word, tag in zip(words, predicted_tags):
                    str_r += f'{word}/{tag} '
                str_r = str_r.rstrip() + "\n"
                outfile.write(str_r)


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
    start = time.perf_counter()
    predict_tags(input_f_name, output_f_name, opts)
    end = time.perf_counter()
    report_accuracy(opts)
    print('Elapsed time: ', (end - start))
