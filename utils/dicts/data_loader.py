""" dataloader """

pad = [r"<PAD>"]
initial = [r'f', r'h', r'k', r'l', r'm', r'n', r'p', r's', r't', r'tʰ', r't͡ɕ', r'v', r'x', r'z', r'ŋ', r'ɓ', r'ɗ', r'ɣ', r'ɲ']
medial = [r"w"]
nucleus = [r'a-0', r'a-1', r'a-2', r'a-3', r'a-4', r'a-5', r'aː-0', r'aː-1', r'aː-2', r'aː-3', r'aː-4', r'aː-5', r'e-0', r'e-1', r'e-2', r'e-3', r'e-4', r'e-5', r'eaː-0', r'eaː-1', r'eaː-2', r'eaː-3', r'eaː-4', r'eaː-5', r'i-0', r'i-1', r'i-2', r'i-3', r'i-4', r'i-5', r'iə-0', r'iə-1', r'iə-2', r'iə-3', r'iə-4', r'iə-5', r'o-0', r'o-1', r'o-2', r'o-3', r'o-4', r'o-5', r'u-0', r'u-1', r'u-2', r'u-3', r'u-4', r'u-5', r'uə-0', r'uə-1', r'uə-2', r'uə-3', r'uə-4', r'uə-5', r'ɔ-0', r'ɔ-1', r'ɔ-2', r'ɔ-3', r'ɔ-4', r'ɔ-5', r'ə-0', r'ə-1', r'ə-2', r'ə-3', r'ə-4', r'ə-5', r'əː-0', r'əː-1', r'əː-2', r'əː-3', r'əː-4', r'əː-5', r'ɛ-0', r'ɛ-1', r'ɛ-2', r'ɛ-3', r'ɛ-4', r'ɛ-5', r'ɨ-0', r'ɨ-1', r'ɨ-2', r'ɨ-3', r'ɨ-4', r'ɨ-5', r'ɨə-0', r'ɨə-1', r'ɨə-2', r'ɨə-3', r'ɨə-4', r'ɨə-5']
# nucleus = [r"eaː", r"ɔ", r"iə", r"ɨə", "uə", r"ɛ", r"ɨ", r"əː", r"i", r"e", r"aː", r"u", r"o",  r"ə", r"a"]
# tone = [r"-0", r"-1", r"-2", r"-3", r"-4", r"-5"]
ending = ['iz', r'kpz', r'kz', r'k̟z', r'mz', r'nz', r'pz', r'tz', r'uz', r'ŋmz', r'ŋz', r'ŋ̟z']
space = [r"$"]
type_phoneme = [pad, space, initial, medial, nucleus, ending]

class Vocab:
    def __init__(self, vocab_file="phoneme.txt"):
        self.vocab_file = vocab_file
        self.word2index = {"<PAD>": 0}
        self.index2word = {0: "<PAD>"}
        self.index_type_phoneme = {0: 0}

        self.n_words = 1
        self.read_lang()

    def add_word(self, word):
        for i, type_p in enumerate(type_phoneme):
            if word in type_p:
                self.index_type_phoneme[self.n_words] = i
                break

        self.word2index[word] = self.n_words
        self.index2word[self.n_words] = word
        self.n_words += 1

    def read_lang(self):
        print("Reading vocabulary from {}".format(self.vocab_file))
        with open(self.vocab_file, 'r', encoding="utf8") as rf:
            line = rf.readline()
            while line:
                self.add_word(line.replace("\n", ""))
                line = rf.readline()
        print("Vocabulary size is {}".format(self.n_words))