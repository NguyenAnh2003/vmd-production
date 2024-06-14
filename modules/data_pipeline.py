import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from omegaconf import OmegaConf, DictConfig  # later


class DataProcessingPipeline:
    def __init__(self) -> None:
        self.pretrained_model = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
        self.sample_rate = 16000
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
        audio_array = torch.FloatTensor(audio_array)

        if rate != self.sample_rate:
            audio_array = torchaudio.functional.resample(
                waveform=audio_array, orig_freq=rate, new_freq=self.sample_rate
            )

        return audio_array

    def get_feature(self, input):
        # input is audio file (byte)
        audio_array = self._load_audio_array(input=input)

        audio_feature = self.feature_extractor(
            audio_array, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_values.squeeze(0)

        with torch.no_grad():
            out = self.model(audio_feature)

        return out.last_hidden_state
