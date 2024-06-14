from libs.libs_func import load_config
from modules.data_pipeline import DataProcessingPipeline
from modules.model_modules import ModelModules
from libs.libs_func import _display_word_mispronounce, _compare_transcript_canonical

# config
conf = load_config("./configs/default.yaml")

# init model modules
model_modules = ModelModules(config=conf)
model = model_modules.model # model
vocab = model_modules.vocab # vocab
device = model_modules.device # device

# data pipeline
data_pipeline = DataProcessingPipeline(conf=conf)


# service class
def vmd_service(media, text):
    phonetic_emb, canonical_phoneme = data_pipeline.get_processed_input(
        media, text
    )  # get phonetic embedding and canonical phoneme

    prediction = model.predict(model, vocab, device, phonetic_emb, canonical_phoneme)

    canonical_phoneme = canonical_phoneme.split()  # split to List

    compared_list = _compare_transcript_canonical(canonical_phoneme, prediction)
    compared_result = _display_word_mispronounce(canonical_phoneme, compared_list)
    text = text.split()  # split target text to align with result

    result = dict(zip(text, compared_result))

    return result
