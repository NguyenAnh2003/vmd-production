import torch
from model.vmd_model import VMDModel
import json
from omegaconf import OmegaConf, DictConfig

class ModelModules:
    def __init__(self, config: DictConfig) -> None:
        self.config = OmegaConf.create(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = self.get_vocab() # get vocab
        self.model = self.get_vmd_model()

    def get_vocab(self):
        with open(self.config.vocab_dir, 'r', encoding="utf8") as file:
            vocab = json.load(file)
            return vocab

    def get_vmd_model(self):
        model = VMDModel(self.config.vmd_model)
        model.load_state_dict(torch.load(self.config.service.checkpoint_dir))
        model.to(self.device)
        model.eval()

        return model
