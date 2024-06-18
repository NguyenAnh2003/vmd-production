def _split_charac(arg: str):
    VIETNAMESE_CHARACTER_UPPER = (
        "["
        + "|".join(
            [
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
                "ỲÝỶỸỴ",
            ]
        )
        + "]"
    )
    VIETNAMESE_CHARACTER_LOWER = VIETNAMESE_CHARACTER_UPPER.lower()
    CHARACTER = (
        "[" + VIETNAMESE_CHARACTER_UPPER[1:-1] + VIETNAMESE_CHARACTER_LOWER[1:-1] + "]"
    )


def word2subword(text: str):
    text = " ".join(text.lower().split())
    list_text = text.split()

    vocabs = []
    for text in list_text:
        vocabs += _split_charac(text)
    return [r for r in vocabs if r is not None]


def _word2phoneme(text: str):
    global vocab
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
