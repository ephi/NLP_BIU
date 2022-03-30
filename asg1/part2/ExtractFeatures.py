import sys
from part1.utils import create_word_signature


def build_word_dict_from_input(input_f_name: str):
    word_dic = {}

    with open(input_f_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            for s in line.strip().split(" "):
                word = s.rsplit('/', 1)[0]
                word_dic[word] = word_dic.get(word, 0) + 1
    return word_dic


g_word_dict = {}


def is_rare(word, word_dict):
    return word_dict.get(word, 0) <= 3


def extract(word, prev_pairs_list, future_words_list, word_dict):
    return {
        'w': word if not is_rare(word, word_dict) else '',
        'pw': prev_pairs_list[0][0] if not is_rare(prev_pairs_list[0][0], word_dict) else '',
        'pt': prev_pairs_list[0][1],
        'ppw': prev_pairs_list[1][0] if not is_rare(prev_pairs_list[1][0], word_dict) else '',
        'ppt': prev_pairs_list[1][1],
        'fw': future_words_list[0][0] if not is_rare(future_words_list[0][0], word_dict) else '',
        'ffw': future_words_list[1][0] if not is_rare(future_words_list[1][0], word_dict) else '',
        'ws': create_word_signature(word),
        'p1': word[:1],
        's1': word[-1:],
        'p2': word[:2] if len(word) >= 2 else '',
        's2': word[-2:] if len(word) >= 2 else '',
        'p3': word[:3] if len(word) >= 3 else '',
        's3': word[-3:] if len(word) >= 3 else ''
    }


def extract_features(input_f_name, output_f_name):
    with open(input_f_name, "r", encoding="utf-8") as input_file:
        with open(output_f_name, "w") as output_file:
            lines = input_file.readlines()

            for line in lines:
                prev_pair = ('startline', '<S>')
                prev_prev_pair = ('startline', '<S>')
                pairs = [tuple(pair.split('/')) for pair in line.strip().split(' ')]
                pairs += [('endline', '<E>'), ('endline', '<E>')]
                for i, pair in enumerate(pairs[:-2]):
                    features = extract(pair[0], [prev_pair, prev_prev_pair],
                                       [pairs[i + 1][0], pairs[i + 2][0]], g_word_dict)
                    output_file.write(' '.join([key + "=" + val for key, val in features.items()]) + f' {pair[1]}\n')

                    prev_prev_pair = prev_pair
                    prev_pair = pair


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("invalid args, expected: input_file_name features_output_file_name exiting...")
        exit(1)
    input_f_name = sys.argv[1]
    output_f_name = sys.argv[2]
    g_word_dict = build_word_dict_from_input(input_f_name)
    extract_features(input_f_name, output_f_name)
