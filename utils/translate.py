import csv

""" translate text to phoneme vice versa """

# Lấy dữ liệu để so sánh
def get_vocab_from_file(file: str):
    f = open(file, "r", encoding="UTF-8")
    lines = f.readlines()
    result = []
    for x in lines:
        x = x.replace('\n', '')
        result.append(x.split('\n'))
    f.close()

    vocab = []
    for res in result:
        for r in res:
            vocab.append(r.split(' '))

    return vocab


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


# Lấy ra dữ liệu muốn translate
def get_data_to_trans(file: str, types: str):
    filename = open(file, 'r', encoding='UTF-8')

    data = csv.DictReader(filename)

    data_from_col = []

    for col in data:
        data_from_col.append(col[types])

    return data_from_col

def translate(file: str, method: str, types: str):
    data_trans = []
    phoneme_to_trans = get_data_to_trans(file, types)
    for phoneme in phoneme_to_trans:
        if method == "phoneme_to_text":
            trans = phoneme.split(' $ ')
            for i, tran in enumerate(trans):
                if i == 0:
                    data_trans = []
                result = compare_data(tran, method=method)
                if result is None:
                    continue
                data_trans.extend(result[:1])

            print(f"Translate: {' '.join(data_trans)}")

        elif method == "text_to_phoneme":
            trans = phoneme.split(' ')
            for i, tran in enumerate(trans):
                if i == 0:
                    data_trans = []
                result = compare_data(tran, method=method)
                data_trans.extend(result[1:])
                data_trans.extend("$")

            data_trans.pop()
            print(f"Translate: {' '.join(data_trans)}")

my_vocab = get_vocab_from_file("speech_dataset/lexicon_vmd.txt")
translate("speech_dataset/test_phone.csv", method="phoneme_to_text", types="transcript")
# translate("speech_dataset/test_phones.csv", method="phoneme_to_text", types="transcript")
