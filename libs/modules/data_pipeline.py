import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from omegaconf import OmegaConf, DictConfig  # later
from libs.libs_func import word2phoneme


class DataProcessingPipeline:
    def __init__(self, conf: DictConfig) -> None:
        self.conf = OmegaConf.create(conf)
        self.pretrained_model = self.conf.pre_trained_model
        self.sample_rate = self.conf.rate
        self.model, self.feature_extractor = self.get_pretrained_model_extractor()

    def get_pretrained_model_extractor(self):

        model = Wav2Vec2Model.from_pretrained(self.pretrained_model)

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.pretrained_model
        )

        model.eval()

        return model, feature_extractor

    def _load_audio_array(self, input: torch.Tensor):

        audio_array, rate = torchaudio.load(input)
        audio_array = torch.mean(audio_array, dim=0) # take mean when stereo sound

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

        canonical_phoneme = word2phoneme(text.lower())

        return out.last_hidden_state, canonical_phoneme
