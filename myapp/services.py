from libs.libs_func import load_config
from libs.modules.data_pipeline import DataProcessingPipeline
from libs.modules.model_modules import ModelModules
from libs.modules.data_modules import word2subword
# config
conf = load_config("./configs/default.yaml")

# init model modules
model_modules = ModelModules(config=conf)

# data pipeline
data_pipeline = DataProcessingPipeline(conf=conf)


# vmd service
def vmd_service(media, text):
    phonetic_emb, canonical_phoneme, num_phoneme, canonical_subwords = data_pipeline.get_processed_input(
        media, text
    )  # get phonetic embedding and canonical phoneme

    canonical_phoneme = " ".join(canonical_phoneme)
    prediction = model_modules.get_prediction(phonetic_emb, canonical_phoneme)

    compared_result = data_pipeline.post_process_result(canonical_phoneme, prediction, num_phoneme)

    result = dict(zip(canonical_subwords, compared_result))

    return result
