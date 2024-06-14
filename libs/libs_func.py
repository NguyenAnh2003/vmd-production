import argparse
import yaml


def load_config(path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default=path)
    args = parser.parse_args()
    config_path = args.conf
    param = yaml.safe_load(open(config_path, "r", encoding="utf8"))
    return param


gap_penalty = -1
match_award = 1
mismatch_penalty = -1


def zeros(rows, cols):
    # Define an empty list
    retval = []
    # Set up the rows of the matrix
    for x in range(rows):
        # For each row, add an empty list
        retval.append([])
        # Set up the columns in each row
        for y in range(cols):
            # Add a zero to each column in each row
            retval[-1].append(0)
    # Return the matrix of zeros
    return retval


def match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == "<eps>" or beta == "<eps>":
        return gap_penalty
    else:
        return mismatch_penalty


def Align(seq1, seq2):
    # Store length of two sequences
    n = len(seq1)
    m = len(seq2)

    # Generate matrix of zeros to store scores
    score = zeros(m + 1, n + 1)

    # Calculate score table

    # Fill out first column
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i

    # Fill out first row
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j

    # Fill out all other values in the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + match_score(seq1[j - 1], seq2[i - 1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert)

    # Traceback and compute the alignment

    # Create variables to store alignment
    align1 = []
    align2 = []

    # Start from the bottom right cell in matrix
    i = m
    j = n

    # We'll use i and j to keep track of where we are in the matrix, just like above
    while i > 0 and j > 0:  # end touching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i - 1][j - 1]
        score_up = score[i][j - 1]
        score_left = score[i - 1][j]

        # Check to figure out which cell the current score was calculated from,
        # then update i and j to correspond to that cell.
        if score_current == score_diagonal + match_score(seq1[j - 1], seq2[i - 1]):
            align1.append(seq1[j - 1])
            align2.append(seq2[i - 1])
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1.append(seq1[j - 1])
            align2.append("<eps>")
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1.append("<eps>")
            align2.append(seq2[i - 1])
            i -= 1

    # Finish tracing up to the top left cell
    while j > 0:
        align1.append(seq1[j - 1])
        align2.append("<eps>")
        j -= 1
    while i > 0:
        align1.append("<eps>")
        align2.append(seq2[i - 1])
        i -= 1

    # Since we traversed the score matrix from the bottom right, our two sequences will be reversed.
    # These two lines reverse the order of the characters in each sequence.
    align1 = align1[::-1]
    align2 = align2[::-1]

    return (align1, align2)


def _compare_transcript_canonical(canonical, transcript):
    result = []
    can_align, trans_align = Align(canonical, transcript)
    for can, tran in zip(can_align, trans_align):
        if can != "<eps>" and tran == "<eps>":
            result.append(1)
        elif can != "<eps>" and tran != "<eps>":
            if can != tran:
                result.append(1)
            else:
                result.append(0)
    return result


def _display_word_mispronounce(canonicals, list_compare):
    phoneme_each_word_compare = []
    phoneme_one_word_compare = []
    for i, (can, compare) in enumerate(zip(canonicals, list_compare)):
        if can != "$":
            phoneme_one_word_compare.append(compare)
        if can == "$" or i == len(canonicals) - 1:
            phoneme_each_word_compare.append(phoneme_one_word_compare)
            phoneme_one_word_compare = []
    list_result = [
        0 if any(list_one_word) else 1 for list_one_word in phoneme_each_word_compare
    ]
    return list_result


def word2phoneme(word, path = "./storage/lexicon_vmd.txt"):

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
