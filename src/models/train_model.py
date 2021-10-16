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

class Train:
    def __init__(self,save_best_model,default_root_dir,ROOT_CSV = "../../data/csv/",train_csv="mixture_train_mix_clean_callfriend_spanish.csv",valid_csv="mixture_val_mix_clean_callfriend_spanish.csv",test_csv = "mixture_test_mix_clean_callfriend_spanish.csv"):
        self.ROOT_CSV = ROOT_CSV
        self.PATH_CSV_TRAIN = self.ROOT_CSV + train_csv 
        self.PATH_CSV_VALID = self.ROOT_CSV + valid_csv
        self.PATH_CSV_TEST = self.ROOT_CSV + test_csv
        self.PATH_CONFIG = "../config/conf.yml"
        self.df_train = pd.read_csv(self.PATH_CSV_TRAIN)
        self.df_val = pd.read_csv(self.PATH_CSV_VALID)
        self.df_test = pd.read_csv(self.PATH_CSV_TEST)
        self.save_best_model = save_best_model
        self.default_root_dir = default_root_dir
        self.conf = self.open_config()




    def open_config(self):
        with open(self.PATH_CONFIG) as f:
            conf = yaml.safe_load(f)
            conf["main_args"]={"exp_dir":self.save_best_model}
            exp_dir = self.save_best_model
            return conf

    def initialize_neptune(self):
        Epoch = self.conf["training"]["epochs"]

        # Configurarmos el experimento y sus parametros
        experiment_name = "Test_30_seconds_ConvTasnet_26.4_train_2.9_val_7.3_test_sta_"+str(start)+"_end_"+str(end)
        params=self.conf
        tags = ["30_seconds_TEST_Best_Model_lote_start_"+str(start)+"_end_"+str(end)+"_ConvTasnet__train_from_PRETRAINED_best_epoch_"+str(best_epoch)]

        # Definir Logger 
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NjRkMmY2YS04M2EwLTRiMGMtODk1Ny1mMWQxZTA3NGM1NzAifQ==",
            project_name="josearangos/Tg-speech-separation",experiment_name=experiment_name,
            params = params, tags = tags, close_after_fit=False)


    
    def create_dataloaders(self):
        pass

        
    def run(self):
        # Neptune

        #Create datasets train, valid, test

        #Create dataloaders

        # Create and configure model

        # Train model

        # Save model

        # End neptune

        self.initialize_neptune()





if __name__ == '__main__':
    Train().run()