from optm.soft_prompt import *
from optm.reft import run_reft_pipeline
from runs.config import *
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig
import os

# Replace the file logging setup with console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

@hydra.main(config_path="runs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Change working directory to project root
    os.chdir(hydra.utils.get_original_cwd())
    
    # Form config dict with only the required fields
    config_dict = {k: v for k, v in cfg.items()}
    config_dict["model_name"] = cfg.model
    
    if cfg.prompt_tuning:
        logging.info(f"Starting {config_dict['config_id']} prompt tuning...")
        try:
            run_prompt_tuning_pipeline(config_dict, cfg.num_epochs, cfg.lr)
        except Exception as e:
            error_msg = f"Error running {config_dict['config_id']} prompt tuning: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
    
    if cfg.token_tuning:
        logging.info(f"Starting {config_dict['config_id']} token tuning...")
        try:
            run_token_tuning_pipeline(config_dict, cfg.num_epochs, cfg.lr, "cuda", cfg.cap_num)
        except Exception as e:
            error_msg = f"Error running {config_dict['config_id']} token tuning: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            
    if cfg.reft: 
        logging.info(f"Starting {config_dict['config_id']} reft tuning...")
        try:
            run_reft_pipeline(config_dict, cfg.num_epochs, cfg.lr, "cuda", cfg.cap_num)
        except Exception as e:
            error_msg = f"Error running {config_dict['config_id']} reft tuning: {str(e)}"
            print(error_msg)
            logging.error(error_msg)

if __name__ == "__main__":
    main()