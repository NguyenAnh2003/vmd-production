import torch

class ModelModules:
  def __init__(self) -> None:
    self.model = self.get_vmd_model()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
  def get_vmd_model(self):
    