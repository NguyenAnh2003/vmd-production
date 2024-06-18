import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from omegaconf import OmegaConf, DictConfig  # later
from .data_modules import word2phoneme, word2subword


class DataProcessingPipeline:
    def __init__(self, conf: DictConfig) -> None:
        self.conf = OmegaConf.create(conf)
        self.pretrained_model = self.conf.pre_trained_model
        self.sample_rate = self.conf.rate
        self.model, self.feature_extractor = self.get_pretrained_model_extractor()

        # align seq
        self.gap_penalty = -1
        self.match_award = 1
        self.mismatch_penalty = -1

    def get_pretrained_model_extractor(self):

        model = Wav2Vec2Model.from_pretrained(self.pretrained_model)

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.pretrained_model
        )

        model.eval()

        return model, feature_extractor

    def _load_audio_array(self, input: torch.Tensor):

        audio_array, rate = torchaudio.load(input)
        # audio_array = torch.mean(audio_array, dim=0)  # take mean when stereo sound

        if rate != self.sample_rate:
            audio_array = torchaudio.functional.resample(
                waveform=audio_array, orig_freq=rate, new_freq=self.sample_rate
            )

        return audio_array

    def get_processed_input(self, input, text):
        # input is audio file (byte)
        # text - converted to canonical phoneme
        audio_array = self._load_audio_array(input=input)

        audio_feature = self.feature_extractor(
            audio_array, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_values.squeeze(0)

        with torch.no_grad():
            out = self.model(audio_feature)

        canonical_phoneme = self.process_canonical_phoneme(text.lower())

        return out.last_hidden_state, canonical_phoneme

    def _compare_transcript_canonical(self, canonical_phoneme, prediction):
        result = []
        can_align, trans_align = self.align_seq(canonical_phoneme, prediction)
        for can, tran in zip(can_align, trans_align):
            if can != "<eps>" and tran == "<eps>":
                result.append(1)
            elif can != "<eps>" and tran != "<eps>":
                if can != tran:
                    result.append(1)
                else:
                    result.append(0)
        return result

    def post_process_result(self, canonical_phoneme, prediction, num_phoneme):
        compared_result = self._compare_transcript_canonical(
            canonical_phoneme, prediction
        )
        phoneme_each_word_compare = []
        phoneme_one_word_compare = []
        for i, compare in enumerate(compared_result):
            phoneme_one_word_compare.append(compare)
            if (i + 1) in num_phoneme[1:] or i == len(canonical_phoneme) - 1:
                phoneme_each_word_compare.append(phoneme_one_word_compare)
                phoneme_one_word_compare = []
        list_result = [
            1 if any(list_one_word) else 0
            for list_one_word in phoneme_each_word_compare
        ]
        return list_result

    def _convert_word2subword(self, canonical):
        canonical_subword = word2subword(canonical.lower())
        return canonical_subword

    def process_canonical_phoneme(self, canonical):
        canonical_subword = self._convert_word2subword(canonical)
        num_phoneme = [0]
        list_phoneme = []
        for subw in canonical_subword:
            phoneme = word2phoneme(subw)
            num_phoneme.append(num_phoneme[-1] + len(phoneme))

            list_phoneme += phoneme
        return list_phoneme, num_phoneme

    def _zeros(self, rows, cols):
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

    def _match_score(self, alpha, beta):
        if alpha == beta:
            return self.match_award
        elif alpha == "<eps>" or beta == "<eps>":
            return self.gap_penalty
        else:
            return self.mismatch_penalty

    def align_seq(self, seq1, seq2):
        # Store length of two sequences
        n = len(seq1)
        m = len(seq2)

        # Generate matrix of zeros to store scores
        score = self._zeros(m + 1, n + 1)

        # Calculate score table

        # Fill out first column
        for i in range(0, m + 1):
            score[i][0] = self.gap_penalty * i

        # Fill out first row
        for j in range(0, n + 1):
            score[0][j] = self.gap_penalty * j

        # Fill out all other values in the score matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calculate the score by checking the top, left, and diagonal cells
                match = score[i - 1][j - 1] + self._match_score(
                    seq1[j - 1], seq2[i - 1]
                )
                delete = score[i - 1][j] + self.gap_penalty
                insert = score[i][j - 1] + self.gap_penalty
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
            if score_current == score_diagonal + self._match_score(
                seq1[j - 1], seq2[i - 1]
            ):
                align1.append(seq1[j - 1])
                align2.append(seq2[i - 1])
                i -= 1
                j -= 1
            elif score_current == score_up + self.gap_penalty:
                align1.append(seq1[j - 1])
                align2.append("<eps>")
                j -= 1
            elif score_current == score_left + self.gap_penalty:
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
