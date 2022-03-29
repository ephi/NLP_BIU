import sys
import itertools
from utils import create_word_signature


# This object holds information from extra_file, if exists.
class ExtraFileOpts:
    def __init__(self, extra_file_path):
        self.opts = {}
        try:
            with open(extra_file_path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    r = line.strip().split(":")
                    try:
                        self.opts[r[0]] = float(r[1])
                    except Exception as e:
                        self.opts[r[0]] = r[1]
        except Exception as e:
            print("Was not able to read extra_file.txt")

    def get_option(self, name):
        return self.opts.get(name, None)

    def set_option(self, name, value):
        self.opts[name] = value


START_TAG = '<S>'
END_TAG = '<E>'

# counters for MLE
g_tag_dic = {}
g_word_tag_dic = {}
g_vocabulary_size = 0

g_transition_dic = {}
g_number_of_tags = 0

# aux for viterbi algorithm, saving all detected tags per word
g_tag_per_word_dic = {}


def get_q(t1, t2, t3, lambda1, lambda2, lambda3):
    # Perform a weighted linear interpolation in order to compute the estimate for q.
    q3 = g_transition_dic[(t3, t2, t1)] / g_transition_dic[(t3, t2)] \
        if (t3, t2, t1) in g_transition_dic.keys() else 0
    q2 = g_transition_dic[(t2, t1)] / g_transition_dic[(t2,)] \
        if f'{t2} {t1}' in g_transition_dic.keys() else 0
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


def build_ngram_from_input(input_f_name: str, gram_n: int):
    # get tags as ngram
    tags_ngram_dic = {}
    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = f'startline/{START_TAG} startline/{START_TAG} {line.strip()} endline/{END_TAG}'
            splitdta = line.split(" ")
            n = len(splitdta)
            for i in range(0, n - (gram_n - 1), 1):
                proc = splitdta[i:(i + gram_n)]
                key = []
                for dta in proc:
                    r = dta.split('/')
                    key.append(r[-1])
                key = tuple(key)
                tags_ngram_dic[key] = tags_ngram_dic.get(key, 0) + 1
    return tags_ngram_dic


def build_word_tag_dict_from_input(input_f_name: str, as_signature: bool = False):
    word_tag_dic = {}

    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = f'startline/{START_TAG} startline/{START_TAG} {line.strip()} endline/{END_TAG}'
            for s in line.split(" "):
                r = s.split('/')
                try:
                    word, tag = r
                except ValueError:
                    l_r = len(r) - 2
                    word = ''.join([x + '/' for x in itertools.islice(r, 0, l_r)])
                    word += r[-2]
                    tag = r[-1]

                if as_signature:
                    word = create_word_signature(word)
                    # if word == '^UNK':
                    #     continue
                # else:
                #     word = word.lower()
                key = (word, tag)
                word_tag_dic[key] = word_tag_dic.get(key, 0) + 1
    return word_tag_dic


def dataset_input_to_e_mle_q_mle(input_f_name, e_mle_f_name, q_mle_fname):
    word_tag_dict = build_word_tag_dict_from_input(input_f_name)
    word_signature_tag_dict = build_word_tag_dict_from_input(input_f_name, as_signature=True)
    write_e_mle_file(e_mle_f_name, [word_tag_dict, word_signature_tag_dict])

    # get tags as unigrams
    tags_unigram_dic = build_ngram_from_input(input_f_name, 1)
    # get tags as bigrams
    tags_bigram_dic = build_ngram_from_input(input_f_name, 2)
    # get tags as trigrams
    tags_trigram_dic = build_ngram_from_input(input_f_name, 3)
    write_q_mle_file(q_mle_fname, [tags_unigram_dic, tags_bigram_dic, tags_trigram_dic])


def write_e_mle_file(e_mle_f_name, word_tag_dicts):
    with open(e_mle_f_name, 'w', encoding='utf-8') as file:
        str_to_write = ""
        for word_tag_dict in word_tag_dicts:
            for t in word_tag_dict:
                str_to_write += f"{t[0]} {t[1]}\t{word_tag_dict[t]}\n"
        str_to_write = str_to_write.rstrip()
        file.write(str_to_write)


def write_q_mle_file(q_mle_f_name, tags_dics):
    with open(q_mle_f_name, 'w', encoding='utf-8') as file:
        str_to_write = ""
        for tag_dic in tags_dics:
            for t in tag_dic:
                if type(t) is tuple:
                    str_to_write += f"{' '.join(t)}\t{tag_dic[t]}\n"
                else:
                    str_to_write += f"{t}\t{tag_dic[t]}\n"
        str_to_write = str_to_write.rstrip()
        file.write(str_to_write)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("invalid args, expected: input_file_name q.mle e.mle. exiting...")
        exit(1)
    input_f_name = sys.argv[1]
    q_mle_f_name = sys.argv[2]
    e_mle_f_name = sys.argv[3]
    dataset_input_to_e_mle_q_mle(input_f_name, e_mle_f_name, q_mle_f_name)
