import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
from pathlib import Path

# add src/ to PYTHONPATH programmatically
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

@hydra.main(config_path=f"/{ROOT}/models/cvae/configs", config_name="model/3D_cvae", version_base=None)
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Hydra instantiates 
    #the model from _target_
    
    #model = instantiate(cfg)
    #print(f"\n Instantiated model: {model.__class__.__name__}")


if __name__ == "__main__":
    print(ROOT)
    main()
