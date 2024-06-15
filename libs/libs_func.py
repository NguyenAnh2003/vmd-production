import argparse
import yaml

def load_config(path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default=path)
    args = parser.parse_args()
    config_path = args.conf
    param = yaml.safe_load(open(config_path, "r", encoding="utf8"))
    return param

def word2phoneme(word, path = "./libs/storage/lexicon_vmd.txt"):

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