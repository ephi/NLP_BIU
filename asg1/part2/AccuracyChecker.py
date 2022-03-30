import sys


def accuracy_on_file(pred_f_name, expctd_f_name):
    good = 0
    bad = 0
    with open(pred_f_name, 'r', encoding='utf-8') as pred_file:
        with open(expctd_f_name, 'r', encoding='utf-8') as expctd_file:
            pred_data = pred_file.read().replace("\n", " ").strip().split(" ")
            expctd_data = expctd_file.read().replace("\n", " ").strip().split(" ")
            p_len = len(pred_data)
            e_len = len(expctd_data)
            if p_len != e_len:
                print(f"miss-match at size between prediction file(={p_len}) and expected file(={e_len})")
                return 0
            for i in range(p_len):
                pred_tag = pred_data[i].split("/")[-1]
                expctd_tag = expctd_data[i].split("/")[-1]
                if pred_tag == expctd_tag:
                    good += 1
                else:
                    bad += 1
    return good / (good + bad)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("invalid args, expected: <prediction_file> <expected_file> exiting...")
        exit(1)
    pred_f_name = sys.argv[1]
    expctd_f_name = sys.argv[2]
    acc = accuracy_on_file(pred_f_name, expctd_f_name)
    print(f"{pred_f_name} vs. {expctd_f_name}, accuracy: {acc}")
