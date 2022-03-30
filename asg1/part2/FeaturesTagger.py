import pickle
import sys
import time
from ExtractFeatures import extract
from part1.GreedyTag import report_accuracy

g_word_dict = {}


def predict_sentence(model, sentence, d):
    sentence += ['endline', 'endline']
    prev_pair = ('startline', '<S>')
    prev_prev_pair = ('startline', '<S>')
    predictions = []

    for i, word in enumerate(sentence[:-2]):
        word_feature = extract(word, [prev_pair, prev_prev_pair], sentence[i:(i+2)], g_word_dict)
        feature_vector = d.transform(word_feature)
        prediction = model.predict(feature_vector)
        predictions.append(prediction)

        prev_prev_pair = prev_pair
        prev_pair = (word, prediction)

    return predictions


if __name__ == "__main__":
    input_file_name = sys.argv[1]
    model_file_name = sys.argv[2]
    feature_map_file = sys.argv[3]
    output_file = sys.argv[4]

    start = time.perf_counter()

    with open(input_file_name, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()

    sentences = [line.strip().split(' ') for line in lines]

    with open(feature_map_file, 'rb') as file:
        v = pickle.load(file)
        g_word_dict = pickle.load(file)

    model = pickle.load(open(model_file_name, 'rb'))

    str_i = ''
    for sentence in sentences:
        predictions = predict_sentence(model, sentence)
        str_i += ' '. join([f'{word}/{tag}' for i, (word, tag) in zip(sentences, predictions)]) + '\n'

    with open(output_file, "w") as out_file:
        out_file.write(str_i)

    report_accuracy({'dev_set': '../data/ass1-tagger-dev', 'pred_file': output_file})
    print("Elapsed time %.2f [m]" % ((time.perf_counter() - start) / 60))
