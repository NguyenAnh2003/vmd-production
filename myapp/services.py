from model.vmd_model import VMDModel
from omegaconf import OmegaConf, DictConfig
from libs.libs_func import load_config
from modules.data_pipeline import DataProcessingPipeline

# config
conf = load_config("./configs/default.yaml")

# init model
model = VMDModel(config=conf["train"])

# data pipeline
data_pipeline = DataProcessingPipeline()


# service class
def vmd_service(media, text):
    audio_feature = data_pipeline.get_feature(media)  # get phonetic embedding
    result = model.predict(audio_feature, text)
    return result
