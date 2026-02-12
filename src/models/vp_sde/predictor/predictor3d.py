import torch
import logging
import torch.nn as nn

log = logging.getLogger(__name__)

class LDPredictor3D:
    def __init__(
        self,
        diffusion_model : nn.Module,
        vae_model : nn.Module,
        predictor_config,
    ):
        self._init_config(predictor_config)

        self.vae_model = vae_model
        self.vae_model.eval()

        self.diffusion_model = diffusion_model
        self.diffusion_model.eval()

    def _init_config(self, predictor_config):

        self.time_star = predictor_config.time_star
        self.pfode = predictor_config.pfode
        self.inference_steps = predictor_config.inference_steps
        self.device = predictor_config.device

        
    def to(self, device):
        self.vae_model.to(device)
        self.diffusion_model.to(device)


    @torch.no_grad()
    def predict(self, dict_ord):
        x = dict_ord["image"].to(self.device)

        #print(x.dtype)
        #print(f"vmin: {x.min().item()}, vmax : {x.max().item()}")
        #output = self.vae_model.predict(x, use_mean_embedding = True)
        #x_r = output.pet_linear
        #z = output.embedding

        #print(f"vmin: {x_r.min().item()}, vmax : {x_r.max().item()}")
        

        z = self.vae_model.encoder(x).embedding
        # z = z.unsqueeze(1)
        z_r = self.diffusion_model.sample_from_x0_t(
            z, self.time_star, steps=self.inference_steps, pfode=self.pfode
        )
        z_r = z_r.squeeze(1)
        x_r = self.vae_model.decoder(z_r, cond_mods = None).reconstruction

        anomaly_maps = (x_r - x).abs()

        dict_pred = {"reconstruction": x_r, "anomaly_map": anomaly_maps, "embedding": z,}

        log.debug(f"[Predictor] Prediction complete for batch of size {x.shape[0]}")
        return dict_pred
