
import json
from types import SimpleNamespace
### manage environment variables 
import sys, os
from pathlib import Path
from argparse import ArgumentParser
### model and dataset class import
import torch
import pandas as pd
from model.EncoderEmbedDecoder import *
from model.EncoderDecoder import *
from trainer.Trainer import Trainer
from data.make_dataset import *
# If GPUs are available, use them, otherwise, set device to CPU and use the number of CPU cores
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = torch.cuda.device_count() if device.type != 'cpu' else os.cpu_count()
# Set default dtype to torch.float64
torch.set_default_dtype(torch.float64)




if __name__ == "__main__":
    parser = ArgumentParser(description="Train a model")
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("--name", type=str, default="model", help="Name of the model")
    args = parser.parse_args()
    name = args.name
    config_path = args.config
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    cfg = SimpleNamespace(**config)
    train_dataset, val_dataset, test_dataset = get_datasets(cfg, device=device)


    model = EncoderEmbedDecoder(cfg).to_device(device)
    trainer = Trainer(model=model,
                        cfg=cfg,
                        writer_dir=name,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset)
    trainer.train()
    with open(f"{name}.json", "w") as f:
        json.dump(cfg.__dict__, f)

    #saving the state dict is more stable than saving the model itself
        torch.save(model.state_dict(), f'{name}.pth')