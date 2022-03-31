import sys
import itertools

START_TAG = "<S>"
END_TAG = "<E>"

# counters for MLE
g_tag_dic = {}
g_word_tag_dic = {}
g_vocabulary_size = 0

g_transition_dic = {}
g_number_of_tags = 0

g_tag_per_word_dic = {}

# serializable dic
g_mle_serial = {}


def get_q(t1, t2, t3, lambda1, lambda2, lambda3):
    # Perform a weighted linear interpolation in order to compute the estimate for q.
    q3 = g_transition_dic[(t3, t2, t1)] / g_transition_dic[(t3, t2)] \
        if (t3, t2, t1) in g_transition_dic.keys() else 0
    q2 = g_transition_dic[(t2, t1)] / g_transition_dic[(t2,)] \
        if (t2, t1) in g_transition_dic.keys() else 0
    q1 = g_transition_dic[(t1,)] / g_number_of_tags

    return (lambda3 * q3) + (lambda2 * q2) + (lambda1 * q1)


def get_e(word_tag_tuple, lambda1=None):
    if lambda1 is None:
        lambda1 = 0.35
    word = word_tag_tuple[0]
    # if word[0] != '^':
    #     word = word.lower()
    tag = word_tag_tuple[1]

    key = (word, tag)
    word_tag_value = g_word_tag_dic.get(key, 0)

    return (word_tag_value + lambda1) / (g_tag_dic[tag] + g_vocabulary_size * lambda1)


def deserialize_counters(dic):
    global g_tag_dic
    global g_word_tag_dic
    global g_vocabulary_size
    global g_tag_per_word_dic
    global g_transition_dic
    global g_number_of_tags
    g_tag_dic = dic["g_tag_dic"]
    g_word_tag_dic = dic["g_word_tag_dic"]
    g_vocabulary_size = dic["g_vocabulary_size"]
    g_transition_dic = dic["g_transition_dic"]
    g_number_of_tags = dic["g_number_of_tags"]
    g_tag_per_word_dic = dic["g_tag_per_word_dic"]


def serializae_counters():
    global g_tag_dic
    global g_word_tag_dic
    global g_vocabulary_size
    global g_tag_per_word_dic
    global g_transition_dic
    global g_number_of_tags
    dic = {"g_tag_dic": g_tag_dic, "g_word_tag_dic": g_word_tag_dic, "g_vocabulary_size": g_vocabulary_size,
           "g_transition_dic": g_transition_dic, "g_number_of_tags": g_number_of_tags,
           "g_tag_per_word_dic": g_tag_per_word_dic}
    return dic


def build_transition_counters(q_mle_f_name):
    global g_transition_dic
    global g_number_of_tags
    g_number_of_tags = 0
    g_transition_dic = {}
    with open(q_mle_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            r = line.strip().split("\t")
            value = int(r[1])
            tags = r[0].split(" ")
            if len(tags) == 1:
                g_number_of_tags += value
            g_transition_dic[tuple(tags)] = value


def build_emission_counters(e_mle_f_name):
    global g_tag_dic
    global g_word_tag_dic
    global g_vocabulary_size
    global g_tag_per_word_dic
    g_tag_dic = {}
    g_word_tag_dic = {}
    g_tag_per_word_dic = {}
    g_vocabulary_size = 0
    with open(e_mle_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            r = line.strip().split("\t")
            count = int(r[1])
            words = r[0].split(" ")
            tag = words[1]
            word = words[0]

            word_tag_set = g_tag_per_word_dic.get(word, set())
            word_tag_set.add(tag)
            g_tag_per_word_dic[word] = word_tag_set

            key = tuple([word, tag])
            g_word_tag_dic[key] = count
            g_tag_dic[tag] = g_tag_dic.get(tag, 0) + count
            g_vocabulary_size += 1
