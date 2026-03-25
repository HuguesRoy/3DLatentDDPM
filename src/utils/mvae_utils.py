import os

from glob import glob
from pathlib import Path

def get_model_from_hydra_path(hydra_path, window=None):
    if window is None:
        # If no window is specified, we load the final model
        try:
            model_path = glob(os.path.join(hydra_path, "CVAE*", "final_model"))[0]
        except IndexError:
            try:
                model_path = glob(os.path.join(hydra_path, "RHVAE*", "final_model"))[0]
            except IndexError:
                model_path = glob(os.path.join(hydra_path, "VQVAE*", "final_model"))[0]

    else:
        try:
            model_path = glob(
                os.path.join(
                    hydra_path,
                    "CVAE*",
                    "best_models_per_window",
                    f"checkpoint_epoch_{window}",
                )
            )[0]
        except IndexError:
            model_path = glob(
                os.path.join(
                    hydra_path,
                    "RHVAE*",
                    "best_models_per_window",
                    f"checkpoint_epoch_{window}",
                )
            )[0]
    return Path(model_path) / "model.pt"
