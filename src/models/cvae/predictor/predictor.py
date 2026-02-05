# Define a base CVAE predictor
from torch import nn 
import torch
from multivae.data.utils import DatasetOutput


class CVAEPredictor:
    def __init__(
        self,
        vae_model : nn.Module,
        device = 'cuda'
    ):

        self.vae_model = vae_model
        self.vae_model.eval()

        self.device = device


        
        
    def to(self, device):
        self.vae_model.to(device)


    @torch.no_grad()
    def predict(self, dict_ord):

        x = dict_ord["image"].to(self.device)
        
        input = DatasetOutput(data={"pet_linear":x})
        output = self.vae_model.predict(input, use_mean_embedding = True)
        x_r = output.pet_linear
        z = output.embedding
        anomaly_maps = (x_r - x).abs()

        dict_pred = {"reconstruction": x_r, "anomaly_map": anomaly_maps, "embedding": z,}

        return dict_pred
