import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
from pathlib import Path

# add src/ to PYTHONPATH programmatically
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

@hydra.main(config_path=f"/{ROOT}/models/fanogan/configs/model", config_name="gpgan", version_base=None)
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))

    # Hydra instantiates 
    #the model from _target_
    
    #model = instantiate(cfg)
    #print(f"\n Instantiated model: {model.__class__.__name__}")


if __name__ == "__main__":
    main()
