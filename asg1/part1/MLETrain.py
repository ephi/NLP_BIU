import sys
import itertools


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


# counters for MLE
g_tag_dic = None
g_word_tag_dic = None
g_vocabulary_size = None

g_transition_dic = None
g_number_of_tags = None

# aux for viterbi algorithm, saving all detected tags per word
g_tag_per_word_dic = None


def get_tags(word):
    global g_tag_per_word_dic
    if g_tag_per_word_dic is None:
        print("please call build_emission_counters before calling get_tags")
    w = word.lower()
    return g_tag_per_word_dic.get(w, set())


# in (word,tag)
# out: lidstone smoothed p(word|tag) based on MLE
def get_e(word_tag_tuple, smoothing_lambda=None):
    global g_tag_dic
    global g_word_tag_dic
    if g_tag_dic is None or g_word_tag_dic is None:
        print("please call build_emission_counters with emission (e.mle) file path first.")
        return
    if smoothing_lambda is None:
        smoothing_lambda = 1
    word = word_tag_tuple[0].lower()
    tag = word_tag_tuple[1]
    den = g_tag_dic.get(tag, 0)
    key = (word, tag)
    nem = g_word_tag_dic.get(key, 0)
    # using lidstone smoothing. assume the vocabulary size
    # is the # of unique words in the training data
    return (nem + smoothing_lambda) / (den + (smoothing_lambda * g_vocabulary_size))


def get_q_t1(t1, lambda1=None):
    global g_number_of_tags
    global g_transition_dic
    if g_transition_dic is None or g_number_of_tags is None:
        print("please call build_transition_counters with transition (q.mle) file path first.")
        return
    if lambda1 + (1 - lambda1) != 1:
        print("lambda1 is not in range [0,1]")
        return
    if lambda1 is None:
        lambda1 = 0.01
    key = (t1, None, None)
    # lidstone smooth before end
    p_t1 = (lambda1 + g_transition_dic[key]) / (g_number_of_tags + lambda1 * g_number_of_tags)
    return p_t1


def get_q_t1_given_t2(t1, t2, lambda1=None):
    global g_number_of_tags
    global g_transition_dic
    if g_transition_dic is None or g_number_of_tags is None:
        print("please call build_transition_counters with transition (q.mle) file path first.")
        return
    if lambda1 is None:
        lambda1 = 0.1
    key_den = (t2, None, None)
    key_nem = (t2, t1, None)
    value_nem = g_transition_dic.get(key_nem, 0)
    value_den = g_transition_dic.get(key_den, 0)
    p_t1_given_t2 = 0
    if value_den > 0:
        p_t1_given_t2 = (value_nem / value_den)
    p_t1 = get_q_t1(t1, lambda1)
    p_t1_given_t2 = p_t1_given_t2 * lambda1 + p_t1 * (1 - lambda1)
    return p_t1_given_t2


def get_q_t1_given_t2_t3(t1, t2, t3, lambda1=None, lambda2=None):
    global g_number_of_tags
    global g_transition_dic
    if g_transition_dic is None or g_number_of_tags is None:
        print("please call build_transition_counters with transition (q.mle) file path first.")
        return
    if lambda1 is None:
        lambda1 = 0.01
    if lambda2 is None:
        lambda2 = 0.1
    key_den = (t2, t1, None)
    key_nem = (t3, t2, t1)
    value_den = g_transition_dic.get(key_den, 0)
    value_nem = g_transition_dic.get(key_nem, 0)
    p_t1_given_t2_t3 = 0
    if value_den > 0:
        p_t1_given_t2_t3 = (value_nem / value_den)
    pt1 = get_q_t1(t1, lambda1)
    p_t1_t2 = get_q_t1_given_t2(t1, t2, lambda2)
    p_t1_given_t2_t3 = p_t1_given_t2_t3 * lambda1 + p_t1_t2 * lambda2 + (1 - lambda1 - lambda2) * pt1
    return p_t1_given_t2_t3


# in: tags: t1,t2(opt),t3(opt)
# out: p(t1) if t2 & t3 were not given
#      p(t2|t2) if t3 is not given
#      p(t3|t2,t1)
# all as MLE. using linear interpolation for smoothing:
#   p(t1) = lidstone smoothing
#   p(t1|t2) = lambda1*p(t2) + (1-lambda1)*p(t1|t2)
#   p(t1|t2,t3)=  p(t1|t2,t3)*lambda1+p(t1|t2)*lambda2+p(t3)*(1-lambda1-lambda2)
def get_q(t1, t2=None, t3=None, lambda1=None, lambda2=None):
    if t2 is None and t3 is None:
        return get_q_t1(t1, lambda1)
    if t2 is not None and t3 is None:
        return get_q_t1_given_t2(t1, t2, lambda1)
    return get_q_t1_given_t2_t3(t1, t2, t3, lambda1, lambda2)


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
            key = [None, None, None]
            for i, tag in enumerate(tags):
                key[i] = tag
            if key[1] is None:
                g_number_of_tags += value
            g_transition_dic[tuple(key)] = value


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
            key = []
            words = r[0].split(" ")
            tag = words[1]
            for w in words:
                key.append(w)
                if tag != w:
                    word_tag_set = g_tag_per_word_dic.get(w, set())
                    word_tag_set.add(tag)
                    g_tag_per_word_dic[w] = word_tag_set
            key = tuple(key)
            g_word_tag_dic[key] = count
            g_tag_dic[tag] = g_tag_dic.get(tag, 0) + count
            g_vocabulary_size += 1


def build_ngram_from_input(input_f_name, gram_n):
    # get tags as ngram
    tags_ngram_dic = {}
    with open(input_f_name, 'r', encoding='utf-8') as file:
        fdata = file.read().replace("\n", " ")
        splitdta = fdata.split(" ")
        n = len(splitdta)
        for i in range(0, n - (gram_n - 1), gram_n):
            proc = []
            for k in range(i, i + gram_n):
                proc.append(splitdta[k])
            key = []
            for dta in proc:
                r = dta.split('/')
                key.append(r[-1])
            key = tuple(key)
            tags_ngram_dic[key] = tags_ngram_dic.get(key, 0) + 1
    return tags_ngram_dic


def dataset_input_to_e_mle_q_mle(input_f_name, e_mle_f_name, q_mle_fname):
    word_tag_dic = {}
    tags_unigram_dic = {}

    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            for s in line.strip().split(" "):
                r = s.split('/')
                try:
                    word, tag = r
                except Exception as e:
                    l_r = len(r) - 2
                    word = ''.join([x + '/' for x in itertools.islice(r, 0, l_r)])
                    word += r[-2]
                    tag = r[-1]
                tag = tag
                word = word.lower()
                key = (word, tag)
                word_tag_dic[key] = word_tag_dic.get(key, 0) + 1
                tags_unigram_dic[tag] = tags_unigram_dic.get(tag, 0) + 1
    write_e_mle_file(e_mle_f_name, word_tag_dic)

    # get tags as bigrams
    tags_bigram_dic = build_ngram_from_input(input_f_name, 2)
    # get tags as trigrams
    tags_trigram_dic = build_ngram_from_input(input_f_name, 3)
    write_q_mle_file(q_mle_fname, [tags_unigram_dic, tags_bigram_dic, tags_trigram_dic])


def write_e_mle_file(e_mle_f_name, word_tag_dic):
    with open(e_mle_f_name, 'w', encoding='utf-8') as file:
        str_to_write = ""
        for e in word_tag_dic:
            str_to_write += f"{e[0]} {e[1]}\t{word_tag_dic[e]}\n"
        str_to_write = str_to_write.rstrip()
        file.write(str_to_write)


def write_q_mle_file(q_mle_f_name, tags_dics):
    with open(q_mle_f_name, 'w', encoding='utf-8') as file:
        str_to_write = ""
        for tag_dic in tags_dics:
            for t in tag_dic:
                if type(t) is tuple:
                    str = ""
                    n = len(t)
                    for i in range(0, n - 1):
                        str += f"{t[i]} "
                    str_to_write += str + f"{t[n - 1]}\t{tag_dic[t]}\n"
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
