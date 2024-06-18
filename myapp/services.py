from libs.libs_func import load_config
from libs.modules.data_pipeline import DataProcessingPipeline
from libs.modules.model_modules import ModelModules

# config
conf = load_config("./configs/default.yaml")

# init model modules
model_modules = ModelModules(config=conf)

# data pipeline
data_pipeline = DataProcessingPipeline(conf=conf)


# vmd service
def vmd_service(media, text):
    phonetic_emb, canonical_phoneme, num_phoneme = data_pipeline.get_processed_input(
        media, text
    )  # get phonetic embedding and canonical phoneme
    
    prediction = model_modules.get_prediction(phonetic_emb, canonical_phoneme)

    # canonical_phoneme = canonical_phoneme.split()  # split to List

    compared_result = data_pipeline.post_process_result(canonical_phoneme, prediction, num_phoneme)
    text = text.split()  # split target text to align with result

    result = dict(zip(text, compared_result))

    return result
