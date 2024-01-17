import csv

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

def translate(sentence: str, method: str, my_vocab):
    data_trans = []

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
        trans = sentence.split(' $ ')
        for i, tran in enumerate(trans):
            if i == 0:
                data_trans = []
            result = compare_data(tran, method=method)
            if result is None:
                continue
            data_trans.extend(result[:1])

        print(f"Translate: {' '.join(data_trans)}")

    elif method == "text_to_phoneme":
        trans = sentence.split(' ')
        for i, tran in enumerate(trans):
            if i == 0:
                data_trans = []
            result = compare_data(tran, method=method)
            data_trans.extend(result[1:])
            data_trans.extend("$")

        data_trans.pop()
        print(f"Translate: {' '.join(data_trans)}")

if __name__ == "__main__":
    my_vocab = get_vocab_from_file("dataset/lexicon_vmd.txt")
    phoneme_sequence = "ɗ i-5 uz $ ɗ i-5 ŋ̟z $ ɗ e-5 uz" #
    text = "địu định đệu" #
    translate(text, method="text_to_phoneme") #
    translate(phoneme_sequence, method="phoneme_to_text") #
    # translate("speech_dataset/test_phones.csv", method="phoneme_to_text", types="transcript")
