from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
import yaml
import json
from asteroid import ConvTasNet
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from asteroid.metrics import get_metrics
from asteroid.utils import tensors_to_device
from tqdm import tqdm
from asteroid.dsp.normalization import normalize_estimates

import os
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import soundfile as sf
import torch
import random as random
from asteroid.models import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from asteroid.engine.optimizers import make_optimizer
from pytorch_lightning.loggers.neptune import NeptuneLogger
from ..data.CallSpanish_dataset import CallSpanish
import pandas as pd
from ..config.base_options import BaseOptions

class Train:
    def __init__(self,opt):
        
        self.opt =opt
        self.experiment_name = opt.experiment_name
        self.tags= opt.tags
        self.ROOT_CSV = opt.root_csv
        self.PATH_CSV_TRAIN = self.root_csv + opt.train_csv 
        self.PATH_CSV_VALID = self.root_csv + opt.valid_csv
        self.PATH_CSV_TEST = self.root_csv + opt.test_csv
        self.PATH_CONFIG = opt.PATH_CONFIG
        self.df_train = pd.read_csv(self.PATH_CSV_TRAIN)
        self.df_val = pd.read_csv(self.PATH_CSV_VALID)
        self.df_test = pd.read_csv(self.PATH_CSV_TEST)
        self.save_best_model = opt.save_best_model
        self.default_root_dir = opt.default_root_dir
        self.conf = self.open_config()
        self.neptune_logger= self.initialize_neptune()
        self.conf["training"]["epochs"] = self.opt.epochs



    def open_config(self):
        with open(self.PATH_CONFIG) as f:
            conf = yaml.safe_load(f)
            conf["main_args"]={"exp_dir":self.save_best_model}
            self.exp_dir = self.save_best_model
            return conf

    def initialize_neptune(self):
        
        # Configurarmos el experimento y sus parametros
        experiment_name = self.experiment_name
        params=self.conf
        tags = [self.tags]

        # Definir Logger 
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NjRkMmY2YS04M2EwLTRiMGMtODk1Ny1mMWQxZTA3NGM1NzAifQ==",
            project_name="josearangos/Tg-speech-separation",experiment_name=experiment_name,
            params = params, tags = tags, close_after_fit=False)

        return neptune_logger
    
    def create_dataloaders(self):
        self.train_set = CallSpanish(
                            csv_path=self.PATH_CSV_TRAIN,
                            task="sep_clean",
                            sample_rate=8000,
                            n_src=2,
                            segment=3
                        )
        self.val_set = CallSpanish(
                        csv_path=self.PATH_CSV_VALID,
                            task="sep_clean",
                            sample_rate=8000,
                            n_src=2,
                            segment=3
                        )

        self.train_loader = DataLoader(self.train_set,shuffle=True,batch_size=self.opt.batch_size, drop_last=True,num_workers=self.opt.num_workers)
        self.val_loader = DataLoader(self.val_set, batch_size=self.opt.batch_size, drop_last=True,num_workers=self.opt.num_workers)
            

    def model_inicialize(self):
        return BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k").cuda()


    def create_configure_model(self):
        model = self.model_inicialize()
        optimizer = make_optimizer(model.parameters(), **self.conf["optim"])

        scheduler = None
        if self.conf["training"]["half_lr"]:
                scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
        gpus = -1 if torch.cuda.is_available() else None
        distributed_backend = "ddp" if torch.cuda.is_available() else None

        # Define Loss function.
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

        self.system = System(
                    model=model,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    scheduler=scheduler,
                    config=self.conf,
                )
        # Define callbacks
        callbacks = []
        checkpoint_dir = os.path.join(self.exp_dir, "checkpoints/")
        self.checkpoint = ModelCheckpoint(checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True)

        # Stop Early
        callbacks.append(self.checkpoint)
        if self.conf["training"]["early_stop"]:
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

        self.trainer= pl.Trainer(
                max_epochs=self.conf["training"]["epochs"],
                callbacks=callbacks,
                default_root_dir=self.exp_dir,
                gpus=gpus,
                #distributed_backend=distributed_backend,
                limit_train_batches=1.0,  # Useful for fast experiment
                gradient_clip_val=5.0,
                logger=self.neptune_logger
                
            )

    def save_best_model(self):
        best_k = {k: v.item() for k, v in self.checkpoint.best_k_models.items()}
        with open(os.path.join(self.exp_dir, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)

        state_dict = torch.load(self.checkpoint.best_model_path)
        self.system.load_state_dict(state_dict=state_dict["state_dict"])
        self.system.cpu()

        to_save = self.system.model.serialize()
        best_model_path = os.path.join(self.exp_dir, "best_model.pth")
        torch.save(to_save,best_model_path )

        #Send best model to neptune
        self.neptune_logger.experiment.log_artifact(best_model_path)
                
    def run(self):
        #Create datasets train, valid, test
        #Create dataloaders
        self.create_dataloaders()

        # Create and configure model # Train model
        self.trainer.fit(self.system)
        
        # Save model best model
        self.save_best_model()


        self.neptune_logger.experiment.stop()




if __name__ == '__main__':
    # Read paramatres of command line
    opt = BaseOptions().parse()
    Train(opt).run()