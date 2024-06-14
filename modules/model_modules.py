import torch
from model.vmd_model import VMDModel


class ModelModules:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_vmd_model(self):
        model = VMDModel(self.config["train"])
        model.load_state_dict(torch.load(self.config["service"]["checkpoint_dir"]))
        model.to(self.device)
        model.eval()

        return model
