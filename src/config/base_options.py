import argparse



class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):  
        self.parser.add_argument('--experiment_name', type=str, default='Train convtasnet', help='Experiment name ')
        self.parser.add_argument('--tags', type=str, default='train_test_convtasnet', help='Tags of experiment name')
        self.parser.add_argument('--root_csv', type=str, default='data/csv/', help='CSV located ')
        self.parser.add_argument('--save_best_model', type=str, default='/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/', help='Path save models')
        self.parser.add_argument('--default_root_dir', type=str, default='/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/', help='Path save models')
        self.parser.add_argument('--train_csv', type=str, default='mixture_train_mix_clean_callfriend_spanish.csv', help='csv train model')
        self.parser.add_argument('--valid_csv', type=str, default='mixture_val_mix_clean_callfriend_spanish.csv', help='csv valid model')
        self.parser.add_argument('--test_csv', type=str, default='mixture_test_mix_clean_callfriend_spanish.csv', help='cvs test model')
        self.parser.add_argument('--path_config', type=str, default='src/config/conf.yml', help='path config experiment')
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size to train')
        self.parser.add_argument('--num_workers', type=int, default=4, help='# Workers')
        self.parser.add_argument('--weight_CS', type=int, default=10, help='# weight_CS ')
        self.parser.add_argument('--epochs', type=int, default=100, help='# Epochs to train')

        

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt