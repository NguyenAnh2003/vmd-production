import re

# Tách từ => phụ âm + nguyên âm -> List([PA], [NA]) | List(None, [NA])
def _split_charac(arg: str):
    VIETNAMESE_CHARACTER_UPPER = "[" + "|".join([
        "ABCDEFGHIJKLMNOPQRSTUVXYZ",
        "ÀÁẢÃẠ",
        "ĂẰẮẲẴẶ",
        "ÂẦẤẨẪẬ",
        "Đ",
        "ÈÉẺẼẸ",
        "ÊỀẾỂỄỆ",
        "ÌÍỈĨỊ",
        "ÒÓỎÕỌ",
        "ÔỒỐỔỖỘ",
        "ƠỜỚỞỠỢ",
        "ÙÚỦŨỤ",
        "ƯỪỨỬỮỰ",
        "ỲÝỶỸỴ"
    ]) + "]"
    VIETNAMESE_CHARACTER_LOWER = VIETNAMESE_CHARACTER_UPPER.lower()
    CHARACTER = "[" + VIETNAMESE_CHARACTER_UPPER[1:-1] + VIETNAMESE_CHARACTER_LOWER[1:-1] + "]"

    consonant = [
        r"ngh", r"ch", r"gh", r"gi", r"kh", r"ng",
        r"nh", r"th", r"tr", r"ph", r"qu",
        r"b", r"c", r"d", r"đ", r"g", r"h",
        r"k", r"l", r"m", r"n", r"p", r"q",
        r"r", r"s", r"t", r"v", r"x",
        f"(?={CHARACTER})\w*"
    ]
    consonant = r"(" + "|".join(consonant) + ")"

    list_charact = []
    for charac in re.split(consonant, arg, maxsplit=1):
        if charac != '':
            list_charact.append(charac)

    if len(list_charact) == 1:
        return None, list_charact[0]
    else:
        return list_charact[0], list_charact[1]

# Lấy dữ liệu để so sánh
def _get_vocab_from_file(file: str):
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
            vocab.append(r.split(" ", maxsplit=1))

    return vocab


def word2subword(text: str):
    text = " ".join(text.lower().split())
    list_text = text.split()

    vocabs = []
    for text in list_text:
        vocabs += _split_charac(text)
    return [r for r in vocabs if r is not None]


def _word2phoneme(text: str, path: str = "./libs/storage/dictionary.txt"):
    vocab = _get_vocab_from_file(path)
    rs = None

    str_split = _split_charac(text)
    cons = str_split[0]
    vowel = str_split[1]

    if cons is None:
        for i, sublist in enumerate(vocab):
            if vowel == sublist[0]:
                rs = vocab[i][1:]

    else:
        rs = []
        for i, sublist in enumerate(vocab):
            if cons == sublist[0] or vowel == sublist[0]:
                rs += vocab[i][1:]
    return [r for r in rs if r is not None]


def word2phoneme(text: str):
    text = " ".join(text.split())
    list_text = text.split()
    phonemes = []
    for text in list_text:
        for char in _word2phoneme(text):
            phonemes += char.split()

    return phonemes
