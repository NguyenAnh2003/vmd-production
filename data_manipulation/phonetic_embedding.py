import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def load_model_ASR_VietAI(model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
    """
    Args:
        model_name: Tên model trên HuggingFace, model ASR của VietAI tên là nguyenvulebinh/wav2vec2-base-vietnamese-250h
    Returns:
        None: Nếu model_name trống
        Wav2Vec2FeatureExtractor, Wav2Vec2Model được goi từ model_name
    """
    if model_name:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        return feature_extractor, model
    else:
        return None

feature_extractor, model = load_model_ASR_VietAI()
def phonetic_embedding(audio_array: torch.Tensor, feature_extractor=feature_extractor, model=model):
    """
    @param audio_array: waveform của audio, trích xuất từ torch.load()
    @param padding_value: Gía trị padding
    @return: model.last_hidden_state, embedding phonetic của audio, được trích xuất từ model pretrain ASR VietAI
    """
    inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        model = model(inputs.input_values)
    return model.last_hidden_state.squeeze(0)