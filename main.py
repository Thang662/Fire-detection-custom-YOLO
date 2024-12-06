import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from dotenv import load_dotenv
import os
import comet_ml
from models.yolo_custom.model import CustomYOLO

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base = None, config_path = "configs", config_name = "experimental")
def main(cfg: DictConfig) -> None:
    # Load environment variables and set up cometml
    load_dotenv()
    api_key = os.getenv("API_KEY")
    comet_ml.login(project_name = cfg['train']['project'], api_key = api_key)
    
    # Change to absolute path for data
    cfg['train']['data'] = f"{os.getcwd()}/{cfg['train']['data']}"

    if cfg['model']['loss_func']:
        log.info(f"Changing from default bce loss function to {cfg['model']['loss_func']}")

    # Inititate CustomYOLO model
    model = CustomYOLO(**cfg['model'])

    # Train
    exp_name = f"{cfg['model']['model']}_\
b{cfg['train']['batch']}_\
e{cfg['train']['epochs']}_\
box{cfg['train']['box']}_\
cls{cfg['train']['cls']}_\
dfl{cfg['train']['cls']}_\
{cfg['model']['loss_func']}"
    
    log.info(exp_name)

    exp = comet_ml.start()
    exp.set_name(exp_name)
    model.train(**cfg['train'], name = exp_name)
    log.info(type(model.model))
    model.val()
    exp.end()

if __name__ == "__main__":
    main()