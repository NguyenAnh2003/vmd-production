import argparse
import yaml


def load_config(path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default=path)
    args = parser.parse_args()
    config_path = args.conf
    param = yaml.safe_load(open(config_path, "r", encoding="utf8"))
    return param


def word2phoneme(word, path):

    def _parse_file(file_path):
        result_dict = {}

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 1:
                    key = parts[0]
                    value = parts[1:]
                    result_dict[key] = value

        return result_dict

    vocab = _parse_file(file_path=path)

    words = word.split()
    phonemes = []

    for i, w in enumerate(words):
        if w in vocab:
            phonemes.append(" ".join(vocab[w]))
        else:
            raise "Word not in Vocab"
        if i != len(words) - 1:
            phonemes.append("$")

    return " ".join(phonemes)


def translate(sentence: str, method: str, my_vocab):
    data_trans = []
    try:
        # So sánh dữ liệu
        def compare_data(arg: str, method: str):
            str_split = lambda x: x.split()
            t = str_split(arg)

            for i, sublist in enumerate(my_vocab):
                if t == sublist[1:] and method == "phoneme_to_text":
                    rs = my_vocab[i]
                    return rs
                elif t == sublist[:1] and method == "text_to_phoneme":
                    rs = my_vocab[i]
                    return rs

        if method == "phoneme_to_text":
            trans = sentence.split(" $ ")
            for i, tran in enumerate(trans):
                if i == 0:
                    data_trans = []
                result = compare_data(tran, method=method)
                if result is None:
                    continue
                data_trans.extend(result[:1])
            return data_trans

        elif method == "text_to_phoneme":
            trans = sentence.split(" ")
            for i, tran in enumerate(trans):
                if i == 0:
                    data_trans = []
                result = compare_data(tran, method=method)
                data_trans.extend(result[1:])
                data_trans.extend("$")

            data_trans.pop()
            # print(f"Translate: {' '.join(data_trans)}")
            return " ".join(data_trans)
    except Exception as e:
        print(f"Error: {e}")
        raise e


# Lấy dữ liệu để so sánh
def get_vocab_from_file(file: str):
    f = open(file, "r", encoding="UTF-8")
    lines = f.readlines()
    result = []
    for x in lines:
        x = x.replace("\n", "")
        result.append(x.split("\n"))
    f.close()

    vocab = []
    for res in result:
        for r in res:
            vocab.append(r.split(" "))
    return vocab


if __name__ == "__main__":
    pass
